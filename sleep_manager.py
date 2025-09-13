from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional


class SleepManager:
    """
    Minimal sleep controller:
      - enter when engine._energy <= 0 (or a provided trigger)
      - NREM: consolidation/replay hooks
      - REM: dream/policy hooks
      - energy recharge and return to awake
    """

    def __init__(self, nrem_steps: int = 200, rem_steps: int = 200):
        self.nrem_steps = nrem_steps
        self.rem_steps = rem_steps
        self.state = "awake"

    def maybe_sleep(self, *, energy: float, enter_threshold: float = 0.0) -> bool:
        return energy <= enter_threshold

    def run(self,
            *,
            get_replay: Callable[[], List[Dict[str, Any]]],
            nrem_hook: Callable[[List[Dict[str, Any]]], None],
            rem_hook: Callable[[List[Dict[str, Any]]], None],
            recharge: Callable[[], None],
            on_state: Optional[Callable[[str], None]] = None) -> None:
        self.state = "nrem"
        if on_state:
            try:
                on_state("nrem")
            except Exception:
                pass
        replay = get_replay()
        try:
            steps = max(1, int(self.nrem_steps))
            for _ in range(steps):
                nrem_hook(replay)
        except Exception:
            pass

        self.state = "rem"
        if on_state:
            try:
                on_state("rem")
            except Exception:
                pass
        try:
            steps = max(1, int(self.rem_steps))
            for _ in range(steps):
                rem_hook(replay)
        except Exception:
            pass

        recharge()
        self.state = "awake"
        if on_state:
            try:
                on_state("awake")
            except Exception:
                pass


