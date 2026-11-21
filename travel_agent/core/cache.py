"""Small bounded, thread-safe TTL cache for in-process response reuse."""

from collections import OrderedDict
from threading import RLock
import time
from typing import Generic, TypeVar


T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, max_size: int = 500, ttl_seconds: float = 600.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[object, tuple[float, T]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: object) -> T | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            timestamp, value = item
            if time.monotonic() - timestamp >= self.ttl_seconds:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: object, value: T) -> None:
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)
