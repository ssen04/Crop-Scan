"""
code/gan_ndvi/flux_trainer.py
Experimental: fine-tune FLUX.1 for RGB → NDVI using rectified flow.

NOTE: FLUX.1 requires significantly more VRAM than ControlNet (SD-1.5).
      Use ControlNet (controlnet_trainer.py) unless you have an H100/A100 80 GB.
      This file is kept for reference and future experiments.

Requires::

    pip install diffusers[torch] transformers accelerate safetensors sentencepiece
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm


class FLUXNDVITrainer:
    """
    Fine-tune FLUX.1 for RGB → NDVI translation using rectified flow matching.

    Advantages over ControlNet (SD-1.5):
      - Rectified flow: faster convergence, more stable gradients
      - Superior detail preservation at high resolution
      - Better handling of agricultural textures
      - State-of-the-art 2024 architecture (black-forest-labs/FLUX.1)

    Args:
        device:        ``'cuda'`` (required — FLUX.1 is very large).
        model_variant: ``'schnell'`` (4-step, fast) or ``'dev'`` (quality).
    """

    def __init__(self, device: str = "cuda", model_variant: str = "schnell"):
        try:
            from diffusers import FluxPipeline
        except ImportError:
            raise ImportError("Install diffusers>=0.28: pip install diffusers[torch]")

        self.device = device
        model_id = f"black-forest-labs/FLUX.1-{model_variant}"

        print(f"Loading FLUX.1-{model_variant}…  (this may take several minutes)")
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe.to(device)

        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer

        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(True)

        self.optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01,
        )

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss: float = float("inf")
        print("✓ FLUX.1 loaded")

    # ------------------------------------------------------------------ #

    def _encode(self, rgb_batch: torch.Tensor,
                ndvi_batch: torch.Tensor):
        """Encode both modalities to VAE latent space."""
        with torch.no_grad():
            rgb_lat = self.vae.encode(rgb_batch).latent_dist.sample()
            ndvi_lat = self.vae.encode(ndvi_batch.repeat(1, 3, 1, 1)).latent_dist.sample()
            scale = self.vae.config.scaling_factor
            return rgb_lat * scale, ndvi_lat * scale

    def train_step(self, rgb_batch: torch.Tensor,
                   ndvi_batch: torch.Tensor) -> float:
        self.transformer.train()
        rgb_batch = rgb_batch.to(self.device)
        ndvi_batch = ndvi_batch.to(self.device)

        rgb_lat, ndvi_lat = self._encode(rgb_batch, ndvi_batch)

        # Rectified flow: t ~ Uniform(0,1)
        t = torch.rand(rgb_lat.shape[0], device=self.device)
        noise = torch.randn_like(ndvi_lat)
        noisy = (1 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * ndvi_lat

        # Transformer predicts velocity field  v = target – noise
        pred = self.transformer(
            hidden_states=noisy,
            timestep=t,
            encoder_hidden_states=rgb_lat,
            return_dict=True,
        ).sample

        loss = F.mse_loss(pred, ndvi_lat - noise)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def validate(self, val_loader) -> float:
        self.transformer.eval()
        total = 0.0
        with torch.no_grad():
            for rgb_batch, ndvi_batch in val_loader:
                rgb_batch = rgb_batch.to(self.device)
                ndvi_batch = ndvi_batch.to(self.device)
                rgb_lat, ndvi_lat = self._encode(rgb_batch, ndvi_batch)
                t = torch.rand(rgb_lat.shape[0], device=self.device)
                noise = torch.randn_like(ndvi_lat)
                noisy = (1 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * ndvi_lat
                pred = self.transformer(
                    hidden_states=noisy, timestep=t,
                    encoder_hidden_states=rgb_lat, return_dict=True
                ).sample
                total += F.mse_loss(pred, ndvi_lat - noise).item()
        return total / len(val_loader)

    def train(self, train_loader, val_loader, num_epochs: int, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nFLUX.1 RGB→NDVI | epochs={num_epochs}")

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            loss_sum = 0.0
            pbar = tqdm(train_loader, desc="Training")
            for rgb, ndvi in pbar:
                l = self.train_step(rgb, ndvi)
                loss_sum += l
                pbar.set_postfix({"loss": f"{l:.4f}"})

            avg = loss_sum / len(train_loader)
            val_loss = self.validate(val_loader)
            self.train_losses.append(avg)
            self.val_losses.append(val_loss)
            print(f"  Train={avg:.4f}  Val={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "transformer_state_dict": self.transformer.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_loss": self.best_val_loss,
                }, os.path.join(save_dir, "best_flux_ndvi.pth"))
                print(f"  ✓ Saved best checkpoint (val={val_loss:.4f})")

        print(f"\n✓ Done — best val loss: {self.best_val_loss:.4f}")
