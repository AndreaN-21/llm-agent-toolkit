
import time
import threading
import itertools
import sys

# ---------------------------------------------------------------------------
# Spinner — animated terminal indicator for long-running API calls
# ---------------------------------------------------------------------------
class Spinner:
    """
    Displays an animated spinner in the terminal while an API call is running.
 
    Usage:
        with Spinner("Thinking..."):
            result = call_api()   # spinner runs until this returns
 
    HOW IT WORKS:
    The spinner runs on a separate daemon thread so it doesn't block the main
    thread that is waiting for the API response
 
    DAEMON THREAD:
    Setting daemon=True means the thread is automatically killed if the main
    program exits — it never keeps the process alive on its own.
 
    CARRIAGE RETURN TRICK:
    Printing \r moves the cursor back to the start of the line without moving
    to the next line, so the next print overwrites the spinner in place.
    This is how the animation works without scrolling the terminal.
    """
 
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
 
    def __init__(self, message: str = "Thinking"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
 
    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            # \r returns to start of line — overwrites previous frame in place
            sys.stdout.write(f"\r  {frame}  {self.message}...")
            sys.stdout.flush()
            time.sleep(0.08)
        # Erase the spinner line completely when done
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()
 
    def __enter__(self):
        self._thread.start()
        return self
 
    def __exit__(self, *_):
        self._stop_event.set()
        self._thread.join()
 
 