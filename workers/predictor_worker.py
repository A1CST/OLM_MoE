from typing import Optional
import torch
import torch.nn as nn
from .template_dlst import TemplateDLSTM


class PredictorWorker(TemplateDLSTM):
    """
    Predictor worker:
    - Input = [packet_64, live_pred_z, hash_bits]
    - Output = next live latent, with residual on live_pred_z (scaled 0.7 to match current DPipe)
    - Loss = MSE to true_next + optional reg to frozen + depth entropy bonus
    """

    def __init__(
        self,
        latent_dim: int,
        hash_bits: int = 64,
        packet_dim: int = 64,
        hidden: int = 256,
        max_layers: int = 3,
        device: Optional[torch.device] = None,
    ):
        in_dim = packet_dim + latent_dim + hash_bits
        out_dim = latent_dim
        super().__init__(in_dim=in_dim, out_dim=out_dim, hidden=hidden, max_layers=max_layers, device=device)
        self.latent_dim = latent_dim
        self.hash_bits = hash_bits
        self.packet_dim = packet_dim
        self._residual_scale = 0.7

    def forward(self, step_inp: torch.Tensor) -> torch.Tensor:
        # Base prediction from template head
        feats = self.forward_features(step_inp)          # (hidden,)
        base = self.head(feats)                          # (latent_dim,)
        # Residual from live_pred_z slice in the last input frame
        s = step_inp[-1, 0, :]                           # (in_dim,)
        live_pred = s[self.packet_dim : self.packet_dim + self.latent_dim]
        y = base + self._residual_scale * live_pred
        return y

    def forward_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward for world predictor evaluation that takes just latent vectors.
        For NREM training, we only have z -> z_next pairs without the full step context.
        Process each item individually since the original forward expects batch_size=1.
        """
        try:
            print(f"DEBUG: forward_latent called with z.shape={z.shape}")
            if z.dim() == 1:
                z = z.unsqueeze(0)  # (latent_dim,) -> (1, latent_dim)
            batch_size, latent_dim = z.shape
            print(f"DEBUG: batch_size={batch_size}, latent_dim={latent_dim}")
            print(f"DEBUG: self.packet_dim={self.packet_dim}, self.latent_dim={self.latent_dim}, self.hash_bits={self.hash_bits}")
            print(f"DEBUG: Expected input dim = {self.packet_dim + self.latent_dim + self.hash_bits}")
            
            # Hard error for latent dimension mismatches during NREM
            if latent_dim != self.latent_dim:
                raise ValueError(f"NREM shape hygiene error: Input latent dimension {latent_dim} != model latent dimension {self.latent_dim}. Fix the pipeline instead of padding.")
            
            # Process each item individually since forward() expects batch_size=1
            results = []
            for i in range(batch_size):
                z_single = z[i:i+1]  # (1, latent_dim) - keep batch dimension
                
                # Create dummy packet and hash inputs for this single item
                packet_dummy = torch.zeros(1, self.packet_dim, dtype=z.dtype, device=z.device)
                hash_dummy = torch.zeros(1, self.hash_bits, dtype=z.dtype, device=z.device)
                
                # Concatenate to match expected input format: [packet, latent, hash]
                step_inp_single = torch.cat([packet_dummy, z_single, hash_dummy], dim=1)  # (1, in_dim)
                step_inp_single = step_inp_single.unsqueeze(0)  # Add sequence dimension: (1, 1, in_dim)
                
                # Use the regular forward method for this single item
                result_single = self.forward(step_inp_single)  # (latent_dim,)
                results.append(result_single)
            
            # Stack all results back into a batch
            if len(results) == 1:
                final_result = results[0]
                print(f"DEBUG: Returning single result: {final_result.shape}")
                return final_result
            else:
                final_result = torch.stack(results, dim=0)  # (batch_size, latent_dim)
                print(f"DEBUG: Returning batched result: {final_result.shape}")
                return final_result
                
        except Exception as e:
            print(f"DEBUG: Exception in forward_latent: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise

    def loss(
        self,
        pred_latent: torch.Tensor,
        true_next_latent: torch.Tensor,
        frozen_latent: Optional[torch.Tensor] = None,
        reg_w: float = 0.1,
        depth_entropy_w: float = 0.001,
    ) -> torch.Tensor:
        mse = torch.mean((pred_latent - true_next_latent) ** 2)
        total = mse
        if frozen_latent is not None and reg_w > 0.0:
            reg = torch.mean((pred_latent - frozen_latent) ** 2)
            total = total + reg_w * reg
        if self._last_depth_weights is not None and depth_entropy_w > 0.0:
            w = self._last_depth_weights + 1e-9
            ent = -torch.sum(w * torch.log(w))
            total = total - depth_entropy_w * ent
        return total

    def optimizer(self, lr: float = 1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))

    def as_world_module(self):
        """
        Return a module/callable mapping z -> z_next.
        Prefers torch.nn.Module if present; else returns a callable.
        """
        for attr in ("model", "net", "module", "core"):
            m = getattr(self, attr, None)
            if m is not None:
                return m
        return self if hasattr(self, "forward") or callable(self) else None

