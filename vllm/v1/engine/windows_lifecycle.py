"""Windows process control only; no Torch, model, sampler, or HTTP implementation."""

import math
import time

DEFAULT_CLEANUP_GRACE_SECONDS = 30.0


class WindowsShutdownError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            "Windows EngineCore cooperative cleanup failed: " + repr(report)
        )


def cleanup_grace(timeout):
    # A zero request-drain timeout means abort requests in original _handle_shutdown.
    # It must not also mean zero time for GPU/Graph/native-worker resource cleanup.
    if timeout is None:
        return DEFAULT_CLEANUP_GRACE_SECONDS
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout < 0
    ):
        raise ValueError("Finite nonnegative cleanup timeout required")
    if timeout == 0:
        return DEFAULT_CLEANUP_GRACE_SECONDS
    return float(timeout)


def shutdown_processes(
    processes, controls, *, timeout=None, force_shutdown, clock=time.monotonic
):
    """Request original child finally, join, and verify its cleanup acknowledgement.

    ``controls`` contains one spawn-context (request, completed) Event pair per
    process. Completion is published only after original EngineCore.shutdown.
    TerminateProcess via the pre-existing fallback is used only after a timeout.
    The helper never reports a forced exit as cooperative success.
    """
    if len(processes) != len(controls):
        raise ValueError("One request/ack pair per EngineCore process required")
    grace = cleanup_grace(timeout)
    deadline = clock() + grace
    for request, _ in controls:
        request.set()
    for process in processes:
        if process.pid is not None:
            process.join(max(0.0, deadline - clock()))
    timed_out = [
        process
        for process in processes
        if process.pid is not None and process.is_alive()
    ]
    report = {
        "status": "cooperative_original_child_shutdown",
        "cleanup_grace_seconds": grace,
        "TerminateProcess_fallback_used": bool(timed_out),
        "pending_request_shutdown_validated": False,
        "processes": [],
    }
    if timed_out:
        try:
            force_shutdown(timed_out, timeout=0)
        except BaseException as error:  # noqa: BLE001 - record failure; never relabel forced exit as success.
            report["fallback_error"] = repr(error)
        report["status"] = "failed_cooperative_shutdown_timeout"
    for process, (_, completed) in zip(processes, controls, strict=True):
        report["processes"].append(
            {
                "pid": process.pid,
                "exitcode": process.exitcode,
                "cleanup_acknowledged": completed.is_set(),
                "alive_after_join_or_fallback": process.is_alive()
                if process.pid is not None
                else False,
            }
        )
    if timed_out or any(
        row["pid"] is None
        or row["exitcode"] != 0
        or row["cleanup_acknowledged"] is not True
        or row["alive_after_join_or_fallback"]
        for row in report["processes"]
    ):
        if not timed_out:
            report["status"] = "failed_child_exit_or_cleanup_ack"
        raise WindowsShutdownError(report)
    return report


def close_client_resources(resources, timeout=None):
    """Keep original socket/task cleanup even when child cleanup failed."""
    primary = None
    try:
        if resources.engine_manager is not None:
            resources.engine_manager.shutdown(timeout=timeout)
    except BaseException as error:  # noqa: BLE001 - preserve the original failure through resource cleanup.
        primary = error
    try:
        # Manager.shutdown is idempotent via its detached original finalizer.
        # BackgroundResources.__call__ then closes the original ZMQ/tasks.
        resources()
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note(
            "Original client background-resource cleanup also failed: " + repr(error)
        )
    if primary is not None:
        raise primary
