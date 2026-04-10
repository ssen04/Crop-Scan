"""
code/gan_ndvi/controlnet_trainer.py
Fine-tunes Stable Diffusion ControlNet for RGB → NDVI translation.

Architecture:
  - ControlNet (lllyasviel/sd-controlnet-canny) — trainable
  - SD-1.5 UNet + VAE + CLIP text encoder       — frozen
  - AdamW8bit (bitsandbytes) + FP16 + autocast  — memory-efficient

Training objective: MSE between predicted noise and sampled noise (latent
diffusion denoising score matching).
"""

import gc
import os

import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


class ControlNetNDVITrainer:
    """
    Fine-tune ControlNet to translate RGB images into NDVI health maps.

    Args:
        device:        ``'cuda'`` or ``'cpu'``.
        learning_rate: AdamW learning rate (default ``1e-5``).
    """

    _MODEL_ID = "runwayml/stable-diffusion-v1-5"
    _CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
    _PROMPT = "NDVI vegetation health map"

    def __init__(self, device: str = "cuda", learning_rate: float = 1e-5):
        self.device = device

        print("Loading Stable Diffusion ControlNet components…")
        self.controlnet = ControlNetModel.from_pretrained(
            self._CONTROLNET_ID, torch_dtype=torch.float16
        ).to(device)
        self.controlnet.requires_grad_(True)

        self.unet = UNet2DConditionModel.from_pretrained(
            self._MODEL_ID, subfolder="unet", torch_dtype=torch.float16
        ).to(device)
        self.unet.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(
            self._MODEL_ID, subfolder="vae", torch_dtype=torch.float16
        ).to(device)
        self.vae.requires_grad_(False)

        self.text_encoder = CLIPTextModel.from_pretrained(
            self._MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(device)
        self.text_encoder.requires_grad_(False)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self._MODEL_ID, subfolder="tokenizer"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self._MODEL_ID, subfolder="scheduler"
        )

        # Use 8-bit AdamW if bitsandbytes is available, else fall back to standard
        if _BNB_AVAILABLE:
            self.optimizer = bnb.optim.AdamW8bit(
                self.controlnet.parameters(),
                lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01,
            )
        else:
            print("bitsandbytes not found — using standard AdamW (higher VRAM usage)")
            self.optimizer = torch.optim.AdamW(
                self.controlnet.parameters(), lr=learning_rate,
            )

        self._text_embeddings = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss: float = float("inf")

        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"✓ Models loaded  (VRAM used: {vram:.2f} GB)")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_text_embeddings(self) -> torch.Tensor:
        """Cache and return CLIP embeddings for the fixed prompt."""
        if self._text_embeddings is None:
            with torch.no_grad():
                tokens = self.tokenizer(
                    self._PROMPT, padding="max_length",
                    max_length=77, truncation=True, return_tensors="pt",
                )
                self._text_embeddings = self.text_encoder(
                    tokens.input_ids.to(self.device)
                )[0]
        return self._text_embeddings

    def _encode_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to VAE latent space."""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            return latents * self.vae.config.scaling_factor

    # ------------------------------------------------------------------ #
    # Train / validate steps
    # ------------------------------------------------------------------ #

    def train_step(self, rgb_batch: torch.Tensor,
                   ndvi_batch: torch.Tensor) -> float:
        self.controlnet.train()

        rgb_batch = rgb_batch.to(self.device, dtype=torch.float16)
        ndvi_batch = ndvi_batch.to(self.device, dtype=torch.float16)

        # Encode NDVI to latent space (3-channel expansion needed for VAE)
        ndvi_latents = self._encode_to_latents(ndvi_batch.repeat(1, 3, 1, 1))

        noise = torch.randn_like(ndvi_latents)
        timesteps = torch.randint(0, 1000, (1,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(ndvi_latents, noise, timesteps)
        text_emb = self._get_text_embeddings()

        with autocast("cuda"):
            down, mid = self.controlnet(
                sample=noisy_latents, timestep=timesteps,
                encoder_hidden_states=text_emb,
                controlnet_cond=rgb_batch, return_dict=False,
            )
            noise_pred = self.unet(
                noisy_latents, timesteps,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
            ).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), 1.0)
        self.optimizer.step()

        # Explicit cache clear to keep VRAM tidy on a single GPU
        del down, mid, noise_pred, noisy_latents, ndvi_latents, noise
        torch.cuda.empty_cache()

        return loss.item()

    def validate(self, val_loader, max_batches: int = 20) -> float:
        self.controlnet.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i, (rgb_batch, ndvi_batch) in enumerate(val_loader):
                if i >= max_batches:
                    break
                rgb_batch = rgb_batch.to(self.device, dtype=torch.float16)
                ndvi_batch = ndvi_batch.to(self.device, dtype=torch.float16)

                ndvi_latents = self._encode_to_latents(ndvi_batch.repeat(1, 3, 1, 1))
                noise = torch.randn_like(ndvi_latents)
                timesteps = torch.randint(0, 1000, (1,), device=self.device).long()
                noisy_latents = self.noise_scheduler.add_noise(ndvi_latents, noise, timesteps)
                text_emb = self._get_text_embeddings()

                with autocast("cuda"):
                    down, mid = self.controlnet(
                        sample=noisy_latents, timestep=timesteps,
                        encoder_hidden_states=text_emb,
                        controlnet_cond=rgb_batch, return_dict=False,
                    )
                    noise_pred = self.unet(
                        noisy_latents, timesteps,
                        encoder_hidden_states=text_emb,
                        down_block_additional_residuals=down,
                        mid_block_additional_residual=mid,
                    ).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                total_loss += loss.item()

        return total_loss / min(max_batches, len(val_loader))

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #

    def train(self, train_loader, val_loader, num_epochs: int, save_dir: str):
        """
        Run the full training loop.

        Args:
            train_loader: DataLoader for training pairs.
            val_loader:   DataLoader for validation pairs.
            num_epochs:   Number of training epochs.
            save_dir:     Directory to save checkpoints.
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"\nControlNet RGB→NDVI  |  epochs={num_epochs}  "
              f"train={len(train_loader)}  val={len(val_loader)}")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}")

            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc="Training")
            for rgb_batch, ndvi_batch in pbar:
                loss = self.train_step(rgb_batch, ndvi_batch)
                epoch_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_train = epoch_loss / len(train_loader)
            self.train_losses.append(avg_train)

            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            print(f"  Train loss : {avg_train:.4f}  |  Val loss : {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "controlnet_state_dict": self.controlnet.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_losses": self.train_losses,
                        "val_losses": self.val_losses,
                        "best_val_loss": self.best_val_loss,
                    },
                    os.path.join(save_dir, "best_controlnet_ndvi.pth"),
                )
                print(f"  ✓ Saved best checkpoint  (val={val_loss:.4f})")

            # Epoch checkpoint
            torch.save(
                {"epoch": epoch, "controlnet_state_dict": self.controlnet.state_dict()},
                os.path.join(save_dir, f"controlnet_epoch_{epoch}.pth"),
            )

        print(f"\n✓ Training complete — best val loss: {self.best_val_loss:.4f}")
