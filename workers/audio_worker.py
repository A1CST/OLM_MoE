from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import math
import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

@dataclass
class AudioWorkerConfig:
    control_dim: int = 32         # dimension exposed to IILSTM
    hidden_dim: int = 64          # internal LSTM hidden
    vocab_size: int = 64          # number of discrete "sound tokens"
    max_ms: int = 400             # clamp duration
    cooldown_ticks: int = 4       # basic anti-spam
    default_gain: float = 1.0     # 0..1 after clamp

class AudioWorker(nn.Module):
    """
    Foundational worker that only proposes {event,id,gain,dur_ms}.
    Events: 0=noop, 1=play, 2=stop
    NO OS OUTPUT. ActionBus clamps/logs only.
    """
    wtype = "audio"

    def __init__(self, name: str = "audio_v1", cfg: Optional[AudioWorkerConfig] = None, device: Optional[str] = None):
        super().__init__()
        self.name = name
        self.cfg = cfg or AudioWorkerConfig()
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")

        # tiny D-LSTM backbone -> control head (for routing) + action heads (for proposal)
        in_dim = getattr(self.cfg, "control_dim", 32)
        hid = self.cfg.hidden_dim
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
        self.head_ctrl   = nn.Linear(hid, self.cfg.control_dim)         # IILSTM-visible control vector
        self.head_event  = nn.Linear(hid, 3)                            # logits: [noop, play, stop]
        self.head_id     = nn.Linear(hid, self.cfg.vocab_size)          # logits over sound tokens
        self.head_gain   = nn.Linear(hid, 1)                            # sigmoid -> 0..1
        self.head_dur    = nn.Linear(hid, 1)                            # softplus -> 0..max_ms

        # ephemeral state
        self._h: Optional[Tuple[torch.Tensor,torch.Tensor]] = None
        self._cooldown_until = -1

        self.to(self.device)

    def reset(self):
        self._h = None
        self._cooldown_until = -1

    @torch.no_grad()
    def forward_once(self, x_ctrl: torch.Tensor) -> Dict[str, Any]:
        """
        x_ctrl: [1,1,control_dim] shaped control input (e.g., projected live_z or prior control context)
        returns: {"ctrl": np.array(control_dim), "proposal": {...}, "raw": {...}}
        """
        if self._h is None:
            h0 = torch.zeros(1, 1, self.cfg.hidden_dim, device=self.device)
            c0 = torch.zeros(1, 1, self.cfg.hidden_dim, device=self.device)
            self._h = (h0, c0)

        out, self._h = self.lstm(x_ctrl, self._h)      # out: [1,1,hid]
        h = out[:, -1, :]                               # [1,hid]

        ctrl = self.head_ctrl(h)                        # [1,control_dim]
        logits_event = self.head_event(h)               # [1,3]
        logits_id = self.head_id(h)                     # [1,vocab]
        gain = torch.sigmoid(self.head_gain(h))         # [1,1] 0..1
        dur  = torch.nn.functional.softplus(self.head_dur(h)) * (self.cfg.max_ms / 6.0)  # ~0..max_ms

        event_idx = int(torch.argmax(logits_event, dim=-1).item())
        sound_id  = int(torch.argmax(logits_id, dim=-1).item())
        gain_f    = float(gain.item())
        dur_ms    = int(min(self.cfg.max_ms, max(1, float(dur.item()))))

        prop: Dict[str, Any] = {"type": "audio", "event": "noop"}
        if event_idx == 1:
            prop = {"type": "audio", "event": "play", "id": sound_id, "gain": gain_f, "dur_ms": dur_ms}
        elif event_idx == 2:
            prop = {"type": "audio", "event": "stop", "id": sound_id}

        return {
            "ctrl": ctrl.squeeze(0).squeeze(0).detach().cpu().numpy().tolist(),
            "proposal": prop,
            "raw": {
                "logits_event": logits_event.detach().cpu().numpy().tolist(),
                "logits_id": logits_id.detach().cpu().numpy().tolist(),
                "gain": gain_f,
                "dur_ms": dur_ms,
            },
        }

    # Convenience used by callers that mirror other workers
    @torch.no_grad()
    def propose(self, tick: int, ctrl_input: np.ndarray) -> Dict[str, Any]:
        """
        ctrl_input: shape [control_dim], caller provides per-tick control embedding like other workers.
        Enforces a simple cooldown to reduce spam when learning starts.
        """
        if tick < self._cooldown_until:
            return {"ctrl": [0.0]*self.cfg.control_dim, "proposal": {"type": "noop"}, "raw": {"cooldown": True}}

        x = torch.tensor(ctrl_input, dtype=torch.float32, device=self.device).view(1,1,-1)
        out = self.forward_once(x)

        # cooldown on non-noop audio to avoid runaway spam at bring-up
        if out["proposal"].get("type") == "audio" and out["proposal"].get("event") in ("play", "stop"):
            self._cooldown_until = tick + self.cfg.cooldown_ticks

        return out