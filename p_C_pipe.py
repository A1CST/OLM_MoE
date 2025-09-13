import os
from typing import Optional, Tuple
import torch
import torch.nn as nn


class PCPipe(nn.Module):
    """P-C pipe: Untrained P-LSTM extracts hidden-state dynamics; a compressor LSTM
    consumes the hidden sequence and produces a 64-dim tanh packet.

    - P-LSTM: parameters are frozen (no grad, eval mode)
    - Compressor: trainable; output goes through Linear->tanh to [-1, 1]
    - Maintains internal hidden states; provide reset_state() to clear
    """

    def __init__(self, latent_dim: int, p_hidden: int = 128, c_hidden: int = 128, device: Optional[torch.device] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device if device is not None else torch.device('cpu')

        input_dim = 3 * latent_dim  # [frozen_z, live_z, live_pred_z]
        self.p_lstm = nn.LSTM(input_size=input_dim, hidden_size=p_hidden, num_layers=1, batch_first=False)
        self.c_lstm = nn.LSTM(input_size=p_hidden, hidden_size=c_hidden, num_layers=1, batch_first=False)
        self.proj = nn.Linear(c_hidden, 64)
        self.tanh = nn.Tanh()

        # Freeze P-LSTM
        for p in self.p_lstm.parameters():
            p.requires_grad = False
        self.p_lstm.eval()

        # Hidden states
        self._p_hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._c_hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_state(self):
        self._p_hc = None
        self._c_hc = None

    def forward(self, seq_steps: torch.Tensor) -> torch.Tensor:
        """seq_steps: (seq_len, batch=1, 3*latent_dim) float tensor on device
        Returns: (64,) packet tensor
        """
        # P-LSTM over sequence
        p_out, self._p_hc = self.p_lstm(seq_steps, self._p_hc)  # p_out: (seq_len, 1, p_hidden)
        # Compressor over P hidden sequence
        c_out, self._c_hc = self.c_lstm(p_out, self._c_hc)  # c_out: (seq_len, 1, c_hidden)
        last = c_out[-1, 0, :]  # (c_hidden,)
        pkt = self.tanh(self.proj(last))  # (64,)
        return pkt

    def optimizer(self, lr: float = 1e-4):
        # Only train compressor and projection
        return torch.optim.Adam(list(self.c_lstm.parameters()) + list(self.proj.parameters()), lr=lr, betas=(0.9, 0.999))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'p_lstm': self.p_lstm.state_dict(),
            'c_lstm': self.c_lstm.state_dict(),
            'proj': self.proj.state_dict(),
        }, path)

    def load(self, path: str, map_location: Optional[torch.device] = None):
        state = torch.load(path, map_location=map_location or self.device)
        self.p_lstm.load_state_dict(state['p_lstm'])
        self.c_lstm.load_state_dict(state['c_lstm'])
        self.proj.load_state_dict(state['proj'])


