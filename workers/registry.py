from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class WorkerStats:
    use_rate_ema: float = 0.0
    marginal_gain_ema: float = 0.0
    conflict_rate_ema: float = 0.0
    ema_alpha: float = 0.1

    def update(self, used: bool, marginal_gain: Optional[float] = None, conflicted: Optional[bool] = None):
        if used:
            self.use_rate_ema = (1 - self.ema_alpha) * self.use_rate_ema + self.ema_alpha * 1.0
        else:
            self.use_rate_ema = (1 - self.ema_alpha) * self.use_rate_ema
        if marginal_gain is not None:
            self.marginal_gain_ema = (1 - self.ema_alpha) * self.marginal_gain_ema + self.ema_alpha * float(marginal_gain)
        if conflicted is not None:
            self.conflict_rate_ema = (1 - self.ema_alpha) * self.conflict_rate_ema + self.ema_alpha * (1.0 if conflicted else 0.0)


@dataclass
class WorkerEntry:
    name: str
    wtype: str                          # e.g., "predictor", "motor", "typer"
    handle: Any                         # torch.nn.Module or callable
    control_dim: int = 32
    stats: WorkerStats = field(default_factory=WorkerStats)


class WorkerRegistry:
    def __init__(self):
        self._workers: List[WorkerEntry] = []

    def register(self, name: str, wtype: str, handle: Any, control_dim: int = 32) -> int:
        idx = len(self._workers)
        self._workers.append(WorkerEntry(name=name, wtype=wtype, handle=handle, control_dim=control_dim))
        return idx

    def size(self) -> int:
        return len(self._workers)

    def get(self, idx: int) -> WorkerEntry:
        return self._workers[idx]

    def all(self) -> List[WorkerEntry]:
        return list(self._workers)

    def snapshot(self) -> List[Dict[str, Any]]:
        out = []
        for i, e in enumerate(self._workers):
            out.append({
                "idx": i,
                "name": e.name,
                "wtype": e.wtype,
                "control_dim": e.control_dim,
                "use_rate_ema": e.stats.use_rate_ema,
                "marginal_gain_ema": e.stats.marginal_gain_ema,
                "conflict_rate_ema": e.stats.conflict_rate_ema,
            })
        return out

    def update_stats(self, idx: int, used: bool, marginal_gain: Optional[float] = None, conflicted: Optional[bool] = None):
        self._workers[idx].stats.update(used=used, marginal_gain=marginal_gain, conflicted=conflicted)

    def set_active(self, idx: int, active: bool):
        # no-op placeholder; your IILSTM mask owns activation at runtime
        pass

    def spawn_from(self, parent_idx: int, name: str, handle: Any, control_dim: Optional[int] = None) -> int:
        cd = control_dim if control_dim is not None else self._workers[parent_idx].control_dim
        return self.register(name=name, wtype=self._workers[parent_idx].wtype, handle=handle, control_dim=cd)


