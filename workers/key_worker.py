from typing import Optional
import torch
import torch.nn as nn
from .template_dlst import TemplateDLSTM


class KeyWorker(TemplateDLSTM):
    """
    Input:  [packet_64, control_token_C]
    Output: [logits_K, p_down, p_up, p_noop]
      - top-1 over logits_K chooses key index
      - bus enforces cooldowns and may convert to noop
    """
    def __init__(self, key_count: int = 8, control_dim: int = 32, hidden: int = 256, max_layers: int = 2, device: Optional[torch.device] = None):
        super().__init__(in_dim=64 + control_dim, out_dim=key_count + 3, hidden=hidden, max_layers=max_layers, device=device)
        self.key_count = key_count
        self.control_dim = control_dim
        self.gamma = nn.Linear(control_dim, hidden)
        self.beta  = nn.Linear(control_dim, hidden)
        self.sigmoid = nn.Sigmoid()

    def forward(self, step_inp: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(step_inp)
        s = step_inp[-1, 0, :]
        ctrl = s[64:64+self.control_dim]
        h = feats * torch.tanh(self.gamma(ctrl)) + self.beta(ctrl)
        raw = self.head(h)                               # (K+3,)
        # last 3 are probabilities (down, up, noop)
        out = raw.clone()
        out[self.key_count:] = self.sigmoid(out[self.key_count:])
        return out



