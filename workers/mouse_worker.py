from typing import Optional
import torch
import torch.nn as nn
from .template_dlst import TemplateDLSTM


class MouseWorker(TemplateDLSTM):
    """
    Input:  [packet_64, control_token_C]  -> in_dim = 64 + C
    Output: [dx_hat, dy_hat, p_left, p_right, p_down, p_up]
      - dx,dy will be scaled/clamped by ActionBus
      - p_* are probabilities (sigmoid); bus handles debounce/cooldowns
    """
    def __init__(self, control_dim: int = 32, hidden: int = 256, max_layers: int = 2, device: Optional[torch.device] = None):
        super().__init__(in_dim=64 + control_dim, out_dim=6, hidden=hidden, max_layers=max_layers, device=device)
        self.control_dim = control_dim
        self.gamma = nn.Linear(control_dim, hidden)
        self.beta  = nn.Linear(control_dim, hidden)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, step_inp: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(step_inp)          # (hidden,)
        s = step_inp[-1, 0, :]
        ctrl = s[64:64+self.control_dim]
        h = feats * self.tanh(self.gamma(ctrl)) + self.beta(ctrl)
        raw = self.head(h)                               # (6,)
        # map: dx_hat, dy_hat raw; clicks/down/up as sigmoid
        out = raw.clone()
        out[2:] = self.sigmoid(out[2:])                  # p_left, p_right, p_down, p_up in (0,1)
        return out



