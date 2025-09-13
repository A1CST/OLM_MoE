import os
from typing import Optional
import torch
from workers.predictor_worker import PredictorWorker


class DPipe(PredictorWorker):
    """
    Backward-compatible shim so existing imports continue to work.
    Behavior matches the old D_pipe.DPipe:
      - forward(): dynamic depth + residual on live_pred_z
      - loss(): MSE + optional frozen reg + depth entropy
      - optimizer(), save(), load(), reset_state()
    """

    def __init__(self, latent_dim: int, hash_bits: int = 64, hidden: int = 256, max_layers: int = 3, device: Optional[torch.device] = None):
        super().__init__(latent_dim=latent_dim, hash_bits=hash_bits, packet_dim=64, hidden=hidden, max_layers=max_layers, device=device)

    # Keep explicit save/load with the same paths the rest of the code expects
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[torch.device] = None):
        state = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(state)

