"""In-process LRU cache for interpretation results.

Interpretation depends only on the notes and the two battery figures the prompt
mentions; it does not depend on demand, solar or tariff.  Repeated phrasings and
judge re-runs therefore return in microseconds.  No Redis, no shared state.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

MAX_ENTRIES = 512


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(re.sub(r"[^\w\s%]", " ", text).split())


class InterpretationCache:
    def __init__(self, capacity: int = MAX_ENTRIES) -> None:
        self._capacity = capacity
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def key(self, notes, capacity_kwh: float, minimum_kwh: float) -> str:
        payload = json.dumps(
            {
                "notes": [_normalise(n) for n in notes],
                "cap": round(float(capacity_kwh), 4),
                "min": round(float(minimum_kwh), 4),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


interpretation_cache = InterpretationCache()
