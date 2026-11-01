"""Opt-in worker-owned scalar CUDA memory observations, outside request timing.

No Tensor creation/copy, CUDA initialization, synchronization, peak reset or
model execution. This does not measure total UMA/Windows physical memory.
"""
import functools
import json
import os
from pathlib import Path
import threading
import time
import uuid


def _output_directory():
    value = os.environ.get("MBX_WORKER_MEMORY_REPORT_DIR")
    if not value:
        return None
    root = next((parent for parent in Path(__file__).resolve().parents
                 if (parent / "sources.lock.json").is_file() and (parent / ".venv").is_dir()), None)
    if root is None:
        raise RuntimeError("Cannot identify task-owned memory observation root")
    directory = Path(value).resolve()
    if not directory.is_relative_to(root / "results"):
        raise RuntimeError("Worker memory records must stay inside owned task results")
    return directory


def snapshot(owner, phase, *, error=None, torch_module=None):
    record = dict(status="worker_memory_observation_disabled", phase=str(phase), pid=os.getpid(),
        thread_id=threading.get_ident(), monotonic_seconds=time.monotonic(),
        tensor_copies=False, Tensor_allocations=False, synchronization_added=False,
        peak_reset_added=False, physical_UMA_measurement=False,
        peak_scope="Ambient original Torch peak counters; original engine may reset them during profiling")
    if error is not None:
        record["observed_failure"] = repr(error)
    try:
        directory = _output_directory()
        if directory is None:
            return record
        if torch_module is None:
            import torch as torch_module
        record["CUDA_initialized"] = bool(torch_module.cuda.is_initialized())
        worker = vars(owner).get("worker", owner)
        record["owner_class"] = type(worker).__module__ + "." + type(worker).__qualname__
        device = getattr(worker, "device", None)
        record["device"] = str(device)
        record["rank"] = getattr(worker, "rank", None)
        values = {}
        errors = {}
        if record["CUDA_initialized"]:
            for name in ("memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved"):
                try:
                    values[name + "_bytes"] = int(getattr(torch_module.cuda, name)(device))
                except Exception as failure:
                    errors[name] = repr(failure)
            try:
                free, total = torch_module.cuda.mem_get_info(device)
                values.update(device_free_bytes=int(free), device_total_bytes=int(total))
            except Exception as failure:
                errors["mem_get_info"] = repr(failure)
        status = ("worker_CUDA_not_initialized_no_memory_queries" if not record["CUDA_initialized"] else
                  "observed_worker_scalar_memory" if not errors else "partial_worker_memory_observation")
        record.update(status=status,
            values=values, query_errors=errors,
            allocator_scope="This worker process Torch allocator only; excludes some native allocations",
            device_free_scope="Whole CUDA device at query time, including other processes/native allocations")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("worker-" + str(record["pid"]) + "-" + uuid.uuid4().hex + ".json")
        record["path"] = str(path)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    except Exception as failure:
        record.update(status="failed_worker_memory_observation_only", observation_error=repr(failure))
    return record


def memory_boundary(phase):
    def decorate(function):
        @functools.wraps(function)
        def invoke(owner, *args, **kwargs):
            snapshot(owner, phase + ":before")
            try:
                result = function(owner, *args, **kwargs)
            except BaseException as primary:
                snapshot(owner, phase + ":failed", error=primary)
                raise
            snapshot(owner, phase + ":after")
            return result
        return invoke
    return decorate
