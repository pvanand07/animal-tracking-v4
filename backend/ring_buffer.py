"""Ring buffer for rolling preroll frame storage."""
import math
import threading
import time
from collections import deque

import numpy as np


class RingBuffer:
    """
    Fixed-size circular buffer storing raw BGR frames with monotonic timestamps.
    Always full (FIFO eviction). Thread-safe.
    """

    def __init__(self, fps: float = 30.0, preroll_s: float = 3.5):
        self._maxlen = max(1, math.ceil(fps * preroll_s))
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def append(self, frame: np.ndarray) -> None:
        """Store a BGR frame alongside the current monotonic clock value."""
        ts = time.monotonic()
        with self._lock:
            self._buf.append((ts, frame))

    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        """Return an ordered list of (monotonic_ts, frame) copies."""
        with self._lock:
            return [(ts, f.copy()) for ts, f in self._buf]

    # ------------------------------------------------------------------
    @property
    def maxlen(self) -> int:
        return self._maxlen

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
