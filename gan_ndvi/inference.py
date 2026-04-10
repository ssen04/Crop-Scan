"""
code/gan_ndvi/inference.py
ControlNet inference: generate NDVI health maps from RGB images.
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


class ControlNetNDVIInference:
    """
    Load a trained ControlNet checkpoint and generate NDVI maps from RGB images.

    NDVI colour coding of the output:
      - 🟢 Green  → high chlorophyll / healthy vegetation (NDVI high)
      - 🟡 Yellow → stressed / partially diseased tissue   (NDVI mid)
      - 🔴 Red    → severely diseased or dead tissue        (NDVI low)

    Args:
        checkpoint_path: Path to a ``.pth`` checkpoint saved by
                         ``ControlNetNDVITrainer``.
        device:          ``'cuda'`` or ``'cpu'``.
    """

    _MODEL_ID = "runwayml/stable-diffusion-v1-5"
    _CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
    _PROMPT = "NDVI vegetation health map"

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device

        self.controlnet = ControlNetModel.from_pretrained(
            self._CONTROLNET_ID, torch_dtype=torch.float16
        ).to(device)

        self.unet = UNet2DConditionModel.from_pretrained(
            self._MODEL_ID, subfolder="unet", torch_dtype=torch.float16
        ).to(device)

        self.vae = AutoencoderKL.from_pretrained(
            self._MODEL_ID, subfolder="vae", torch_dtype=torch.float16
        ).to(device)

        self.text_encoder = CLIPTextModel.from_pretrained(
            self._MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self._MODEL_ID, subfolder="tokenizer"
        )

        # Use DDIM for faster inference (30 steps instead of 1000)
        self.scheduler = DDIMScheduler.from_config(
            DDPMScheduler.from_pretrained(self._MODEL_ID, subfolder="scheduler").config
        )

        # Load fine-tuned ControlNet weights
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.controlnet.load_state_dict(ckpt["controlnet_state_dict"])

        self.controlnet.eval()
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        # Preprocessing transform (must match training normalisation)
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

        print("✓ ControlNet NDVI model loaded")

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_ndvi(self, rgb_image_path: str,
                      num_inference_steps: int = 30) -> Image.Image:
        """
        Generate a colourised NDVI health map from an RGB crop image.

        Args:
            rgb_image_path:      Path to the input RGB image.
            num_inference_steps: DDIM denoising steps (fewer = faster, more = sharper).

        Returns:
            PIL Image (RGB) with RdYlGn colourmap applied to the NDVI values.
        """
        # 1. Preprocess RGB
        rgb = Image.open(rgb_image_path).convert("RGB")
        rgb_tensor = self.transform(rgb).unsqueeze(0).to(self.device).half()

        # 2. Text conditioning
        text_inputs = self.tokenizer(
            self._PROMPT, padding="max_length",
            max_length=77, truncation=True, return_tensors="pt",
        )
        text_embeddings = self.text_encoder(
            text_inputs.input_ids.to(self.device)
        )[0]

        # 3. Start from random latent noise
        latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=torch.float16)
        self.scheduler.set_timesteps(num_inference_steps)

        # 4. Iterative DDIM denoising
        for t in self.scheduler.timesteps:
            down, mid = self.controlnet(
                sample=latents, timestep=t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=rgb_tensor, return_dict=False,
            )
            noise_pred = self.unet(
                latents, t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
            ).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Decode latents → pixel space
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        image = (image + 1.0) / 2.0          # [-1,1] → [0,1]
        image = image.clamp(0, 1)

        # 6. Average channels → single NDVI scalar map
        ndvi_gray = image.mean(dim=1)[0].cpu().numpy()   # [H, W]  in [0, 1]

        # 7. Apply RdYlGn colourmap for intuitive visualisation
        from matplotlib import cm
        ndvi_rgb = (cm.get_cmap("RdYlGn")(ndvi_gray)[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(ndvi_rgb)
