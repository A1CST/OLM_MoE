from typing import Optional
import torch
import torch.nn as nn


class IILSTM(nn.Module):
    """
    Internal-Intelligence LSTM (executive):
    Inputs:  packet (64), drivers (D)
    Outputs:
      - routing_logits: (M,) scores over workers
      - k_logits: (Kmax+1,) distribution over how many workers to run
      - control_tokens: (M, C) per-worker control vectors
    """
    def __init__(self, drivers_dim: int, num_workers: int, control_dim: int = 32, hidden: int = 128, kmax: int = 3, device: Optional[torch.device] = None):
        super().__init__()
        self.packet_dim = 64
        self.drivers_dim = drivers_dim
        self.num_workers = num_workers
        self.control_dim = control_dim
        self.hidden = hidden
        self.kmax = kmax
        self.device = device if device is not None else torch.device('cpu')

        in_dim = self.packet_dim + self.drivers_dim
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=False)
        self._hc = None

        self.routing_head = nn.Linear(hidden, num_workers)
        self.k_head = nn.Linear(hidden, kmax + 1)
        self.ctrl_head = nn.Linear(hidden, num_workers * control_dim)
        self.tanh = nn.Tanh()
        self.register_buffer("active_mask", torch.ones(num_workers, dtype=torch.bool), persistent=False)

    def reset_state(self):
        self._hc = None

    def set_active_mask(self, mask: torch.Tensor):
        # mask: (M,) bool; inactive logits get -inf so they aren't selected
        self.active_mask = mask.to(dtype=torch.bool, device=self.routing_head.weight.device)

    def forward(self, packet_64: torch.Tensor, drivers: torch.Tensor):
        """
        packet_64: (64,) float
        drivers  : (D,) float
        Returns: routing_logits (M,), k_logits (Kmax+1,), control_tokens (M, C)
        """
        x = torch.cat([packet_64, drivers], dim=0).view(1, 1, -1)   # (seq=1, batch=1, in_dim)
        out, self._hc = self.lstm(x, self._hc)                      # (1, 1, hidden)
        h = out[-1, 0, :]                                           # (hidden,)
        routing_logits = self.routing_head(h)                       # (M,)
        if self.active_mask is not None and self.active_mask.shape[0] == routing_logits.shape[0]:
            neg_inf = torch.finfo(routing_logits.dtype).min
            routing_logits = torch.where(self.active_mask, routing_logits, torch.full_like(routing_logits, neg_inf))
        k_logits = self.k_head(h)                                   # (Kmax+1,)
        ctrl = self.ctrl_head(h).view(self.num_workers, self.control_dim)
        control_tokens = self.tanh(ctrl)                            # clamp to [-1, 1]
        return routing_logits, k_logits, control_tokens


