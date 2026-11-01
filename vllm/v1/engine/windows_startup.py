"""Windows startup liveness/deadline and failure cleanup; no model/transport math."""

import math
import time


class WindowsStartupWait:
    """Windows HANDLEs cannot be polled as ZMQ file descriptors.

    Recheck owned child exit codes before/after short original socket polls.
    Keep the original configured ready timeout as a total monotonic budget.
    """

    def __init__(self, timeout_seconds, *, clock=time.monotonic):
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("Finite nonnegative original startup timeout required")
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self.deadline = clock() + timeout_seconds

    def check_children(self, manager, coordinator=None, watched=()):
        finished = {}
        get_finished = getattr(manager, "finished_procs", None)
        if get_finished is not None:
            finished.update(get_finished())
        if coordinator is not None and coordinator.exitcode is not None:
            finished[coordinator.name] = coordinator.exitcode
        frontend = {process.name: process.exitcode for process in watched if process.exitcode is not None}
        if finished or frontend:
            raise RuntimeError(
                "Engine core initialization failed. See root cause above. "
                f"Failed core proc(s): {finished}"
                + (f", failed frontend proc(s): {frontend}" if frontend else "")
            )

    def poll(self, poller, timeout_ms, *, manager=None, coordinator=None, watched=()):
        self.check_children(manager, coordinator, watched)
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for engine core processes to start. Waited {self.timeout_seconds}s (configured by VLLM_ENGINE_READY_TIMEOUT_S).")
        events = poller.poll(min(timeout_ms, 1000, max(1, math.ceil(remaining * 1000))))
        self.check_children(manager, coordinator, watched)
        if not events and self.clock() >= self.deadline:
            raise TimeoutError(f"Timed out waiting for engine core processes to start. Waited {self.timeout_seconds}s (configured by VLLM_ENGINE_READY_TIMEOUT_S).")
        return events


def cleanup_failed_client(finalizer, resources, *, primary, close_resources=None):
    """Close owned sockets after failed child startup without masking its error."""
    if finalizer.detach() is None:
        return
    if close_resources is None:
        from vllm.v1.engine.windows_lifecycle import close_client_resources
        close_resources = close_client_resources
    try:
        close_resources(resources)
    except BaseException as error:
        if primary is None:
            raise
        primary.add_note("Original Windows API startup cleanup also failed: " + repr(error))
