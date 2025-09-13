from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


class TemplateDLSTM(nn.Module):
    """
    Generic dynamic-depth D-LSTM worker.

    - Input per step: (seq_len>=1, batch=1, in_dim)
    - LSTM stack with depth gating over layer outputs.
    - Output head is a Linear to out_dim; subclasses decide how to postprocess (e.g., residual).
    - Exposes _last_depth_weights for diagnostics/regularization.

    Use reset_state() between episodes to clear hidden states.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 256,
        max_layers: int = 3,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.max_layers = max_layers
        self.device = device if device is not None else torch.device('cpu')

        self.layers = nn.ModuleList([
            nn.LSTM(input_size=in_dim if i == 0 else hidden, hidden_size=hidden, num_layers=1, batch_first=False)
            for i in range(max_layers)
        ])
        self.depth_gate = nn.Linear(in_dim, max_layers)
        self.head = nn.Linear(hidden, out_dim)

        self._hc: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None for _ in range(max_layers)]
        self._last_depth_weights = None

    def reset_state(self):
        self._hc = [None for _ in range(self.max_layers)]

    def forward_features(self, step_inp: torch.Tensor) -> torch.Tensor:
        """
        Returns the depth-weighted hidden feature before the output head.
        step_inp: (seq_len>=1, 1, in_dim)
        """
        x = step_inp
        layer_outputs = []
        for i, lstm in enumerate(self.layers):
            out, self._hc[i] = lstm(x, self._hc[i])   # out: (seq_len, 1, hidden)
            layer_outputs.append(out[-1, 0, :])       # keep last step feature
            x = out

        s = step_inp[-1, 0, :]                        # last input frame
        weights = torch.softmax(self.depth_gate(s), dim=0)  # (max_layers,)
        self._last_depth_weights = weights
        combined = sum(w * o for w, o in zip(weights, layer_outputs))  # (hidden,)
        return combined

    def forward(self, step_inp: torch.Tensor) -> torch.Tensor:
        """
        Default head application. Subclasses can override to add residuals, etc.
        """
        feats = self.forward_features(step_inp)       # (hidden,)
        y = self.head(feats)                          # (out_dim,)
        return y

    # Optional generic regularized loss (MSE + depth entropy bonus)
    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        depth_entropy_w: float = 0.0,
        mse_w: float = 1.0,
    ) -> torch.Tensor:
        total = torch.mean((pred - target) ** 2) * mse_w
        if self._last_depth_weights is not None and depth_entropy_w > 0.0:
            w = self._last_depth_weights + 1e-9
            ent = -torch.sum(w * torch.log(w))
            total = total - depth_entropy_w * ent
        return total

    def optimizer(self, lr: float = 1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[torch.device] = None):
        state = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(state)


