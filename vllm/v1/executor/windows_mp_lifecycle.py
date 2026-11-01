"""Windows worker ownership only; retain original death-pipe/MQ/RPC transport."""

import threading

WINDOWS_ASYNC_STOP = object()
FAILED_WORKER_OWNERS = []


def shutdown_workers(handles, *, force_shutdown, shutdown_api=None, timeout=None, pending_futures=()):
    """Keep original EOF request, then require successful worker cleanup and join.

    Event request is set before original death_writer closure. The original
    DeathPipeMonitor still wakes the original MQ reader; no new queue consumer,
    CUDA IPC, tensor copy, or model execution is introduced here.
    """
    handles = tuple(handles)
    try:
        if any(not future.done() for future in pending_futures):
            raise RuntimeError("Windows MP idle shutdown cannot drop pending RPC futures")
        controls = []
        for handle in handles:
            request = getattr(handle, "windows_shutdown_request", None)
            completed = getattr(handle, "windows_shutdown_complete", None)
            if request is None or completed is None:
                raise RuntimeError("Require owned Windows request/completion events for every MP worker")
            controls.append((request, completed))
        # Preserve ordering: child may react to EOF immediately.
        for request, _ in controls:
            request.set()
        for handle in handles:
            if handle.death_writer is not None:
                handle.death_writer.close()
                handle.death_writer = None
        if shutdown_api is None:
            from vllm.v1.engine.windows_lifecycle import shutdown_processes
            shutdown_api = shutdown_processes
        report = shutdown_api([handle.proc for handle in handles], controls,
                              timeout=timeout, force_shutdown=force_shutdown)
        return dict(report, request_transport="Original death-pipe EOF / original MQ.shutdown wakeup",
                    worker_cleanup_ack_after_original_shutdown=True,
                    CUDA_IPC_or_tensor_transport_changed=False,
                    scope="Owned local MP workers; idle shutdown only, pending request abort not validated")
    except BaseException:
        # Never close response queues or release owners after unproven shutdown.
        FAILED_WORKER_OWNERS.append((handles, tuple(pending_futures)))
        raise


def cleanup_failed_startup(handles, *, force_shutdown, primary, shutdown_api=None):
    """Retain the original startup error while attaching real cleanup failures."""
    try:
        shutdown_workers(handles, force_shutdown=force_shutdown, shutdown_api=shutdown_api)
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note("Windows MP startup cleanup also failed: " + repr(error))
        return error
    return None


def stop_async_output(worker, *, timeout=30.0):
    """Drain original FIFO output work and drop its last GPU-owner frame.

    The original async thread recognizes the singleton after preceding FIFO
    work. It marks completion only on that path, and the owner joins it before
    original worker/Graph/PLE teardown. This never calls get_output a second time.
    """
    thread = getattr(worker, "async_output_copy_thread", None)
    if thread is None:
        return dict(async_output_thread_present=False, joined=True)
    try:
        if thread is threading.current_thread():
            raise RuntimeError("Async output thread cannot join itself")
        if getattr(worker, "_windows_async_output_drained", False) and not thread.is_alive():
            return dict(async_output_thread_present=True, joined=True, FIFO_stop_observed=True)
        if not thread.is_alive():
            raise RuntimeError("Async output thread exited without the FIFO completion marker")
        worker.async_output_queue.put(WINDOWS_ASYNC_STOP)
        thread.join(timeout)
        if thread.is_alive() or getattr(worker, "_windows_async_output_drained", False) is not True:
            raise RuntimeError("Original async output FIFO has not drained and joined")
        return dict(async_output_thread_present=True, joined=True, FIFO_stop_observed=True)
    except BaseException:
        FAILED_WORKER_OWNERS.append((worker, thread, getattr(worker, "async_output_queue", None)))
        raise
