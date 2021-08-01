import tracemalloc
from time import time

from util.logging import Log


class Measure:
    def __init__(self, label=None, trace_memo=False, indent=1):
        super(Measure, self).__init__()

        self.label = f"{label} - " if label else ""
        self.trace_memo = trace_memo
        self.indent = indent

    def __enter__(self):
        Log.i(f"[EXECUTION METER] {self.label}Watching execution...", indent=self.indent)

        self.start_timestamp = time()

        if self.trace_memo:
            tracemalloc.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time() - self.start_timestamp

        if self.trace_memo:
            current_memory, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            Log.i("[EXECUTION METER] %sExecuted in %.3f seconds (memory=%s, peak=%s)." % (
                self.label, elapsed, Log.readable_size(current_memory), Log.readable_size(peak)
            ), indent=self.indent)
        else:
            Log.i("[EXECUTION METER] %sExecuted in %.3f seconds" % (self.label, elapsed), indent=self.indent)

        if exc_value:
            return False

        return True
