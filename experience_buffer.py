from __future__ import annotations
from typing import Dict, Any, List, Optional


class ExperienceBuffer:
    """
    Online ring buffer for wake data. Push one record per tick.
    Computes reward for the previous record using current drivers.
    Finalized episodes are handed to sleep for training.
    """

    def __init__(self, maxlen: int = 20000):
        self.maxlen = maxlen
        self._pending: Optional[Dict[str, Any]] = None
        self._finalized: List[Dict[str, Any]] = []
        self._episode_id = 0

    def start_episode(self):
        self.flush_pending()
        self._episode_id += 1

    def flush_pending(self):
        if self._pending is not None:
            self._finalized.append(self._pending)
            if len(self._finalized) > self.maxlen:
                self._finalized = self._finalized[-self.maxlen:]
            self._pending = None

    def push(self, rec: Dict[str, Any], reward_fn) -> None:
        """
        rec must include at least: tick, drivers (dict), iilstm_k, iilstm_selected,
        executed_workers, final_action, iilstm logits if available.
        reward_fn(prev_rec, curr_rec) -> float
        """
        if self._pending is not None:
            try:
                self._pending["reward"] = float(reward_fn(self._pending, rec))
            except Exception:
                self._pending["reward"] = None
            self._finalized.append(self._pending)
            if len(self._finalized) > self.maxlen:
                self._finalized = self._finalized[-self.maxlen:]
        self._pending = rec

    def drain(self) -> List[Dict[str, Any]]:
        """Return and clear finalized data (keep the pending last sample)."""
        out = self._finalized
        self._finalized = []
        return out

    # ---- New helpers for world-predictor transitions ----
    def append(self, rec: Dict[str, Any]) -> None:
        """Append a finalized record (already contains next-state fields)."""
        self._finalized.append(rec)
        if len(self._finalized) > self.maxlen:
            self._finalized = self._finalized[-self.maxlen:]

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of finalized records without clearing."""
        return list(self._finalized)


