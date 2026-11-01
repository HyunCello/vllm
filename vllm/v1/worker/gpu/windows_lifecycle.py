"""Release original Windows recipe consumers before original runner cache deletion.

No replacement Graph executor, allocator pool, model math, or PLE implementation.
Imports of real runtime classes happen only inside the shutdown call.
"""

import sys


FAILED_RUNNERS = []


def shutdown_consumers(
    runner,
    torch,
    *,
    manager_type=None,
    standard_type=None,
    breakable_type=None,
    boundary=None,
):
    completed = getattr(runner, "_windows_recipe_consumers_shutdown", None)
    if completed is not None:
        return completed
    if manager_type is None:
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

        manager_type = CudaGraphManager
    if standard_type is None:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        standard_type = CUDAGraphWrapper
    if breakable_type is None:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

        breakable_type = BreakableCUDAGraphWrapper
    if boundary is None:
        from vllm._rtx_spark.runtime import ple as recipe_original_ple_windows

        boundary = recipe_original_ple_windows
    owners = [runner]
    try:
        managers = {
            id(value): value
            for owner in (runner, getattr(runner, "speculator", None))
            if owner is not None
            for value in vars(owner).values()
            if isinstance(value, manager_type)
        }
        standard = list(standard_type._all_instances)
        breakable = list(breakable_type._all_instances)
        wrappers = standard + breakable
        owners.extend((managers, wrappers))
        graphs = {
            id(graph): graph
            for manager in managers.values()
            for graph in manager.graphs.values()
        }
        owners.append(graphs)
        for wrapper in standard:
            for entry in wrapper.concrete_cudagraph_entries.values():
                graph = getattr(entry, "cudagraph", None)
                if graph is not None:
                    graphs[id(graph)] = graph
        for wrapper in breakable:
            for entry in wrapper.entries.values():
                capture = entry.capture
                if capture is None:
                    continue
                if capture._capturing or capture._current_graph is not None:
                    raise RuntimeError("Cannot release an active breakable capture")
                # Original _end_segment appends CUDAGraph.replay; eager breaks
                # are ordinary callables. Retain and reset only actual Graphs.
                for segment in capture.segments:
                    graph = getattr(segment, "__self__", None)
                    if isinstance(graph, torch.cuda.CUDAGraph):
                        if getattr(segment, "__name__", None) != "replay":
                            raise RuntimeError("Unexpected bound CUDAGraph segment")
                        graphs[id(graph)] = graph
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()
        for manager in managers.values():
            manager.release_graphs()
        for wrapper in wrappers:
            wrapper.clear_graphs()
        torch.cuda.synchronize()
        native_cute = {"original_CUDA_libraries_closed": 0, "runtime_not_loaded": True}
        native_cute_runtime = sys.modules.get("original_cute_coff_runtime")
        if native_cute_runtime is not None:
            native_cute = native_cute_runtime.close_after_graphs_drained(
                graphs_reset=True, streams_drained=True
            )
        ple = boundary.shutdown_after_graphs_drained(
            graphs_reset=True, streams_drained=True
        )
        completed = {
            "graphs_reset": len(graphs),
            "original_managers_released": len(managers),
            "original_wrapper_registries_cleared": len(wrappers),
            "all_streams_drained": True,
            "PLE": ple,
            "native_CuTe_AOT": native_cute,
            "before_original_KV_or_model_deletion": True,
            "scope": "original consumer cleanup only; pending-request shutdown not validated",
        }
        runner._windows_recipe_consumers_shutdown = completed
        return completed
    except BaseException:
        # Keep actual Graphs, caches, native PLE owners and runner alive on error.
        # No field clear, false acknowledgement, or retry success is fabricated.
        FAILED_RUNNERS.append(tuple(owners))
        raise
