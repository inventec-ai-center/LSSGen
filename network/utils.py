import cv2
import random
import math
import torch
import torch.fft as fft
import matplotlib.pyplot as plt
import numpy as np

from diffusers import AutoencoderKL

from huggingface_hub import snapshot_download
from pathlib import Path
from PIL import Image, ImageChops, ImageOps, ImageDraw
from torchvision import transforms
from typing import Union, List, Tuple, Optional, Dict, Any

def download_hf_model(repo_id: str, local_dir: str | Path) -> str:
    """
    Download a model from the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID of the model.
        local_dir (str): The local directory where the model should be saved.

    Returns:
        str: The path to the downloaded model.
    """
    if isinstance(local_dir, Path):
        local_dir = local_dir.as_posix()
    return snapshot_download(repo_id=repo_id, local_dir=local_dir)

def expand_mask(image: Image.Image):
    image_np = np.array(image)

    kernel = np.ones((7, 7), np.uint8)
    
    dilated = cv2.dilate(image_np, kernel, iterations=7)
    
    dilated_image = Image.fromarray(dilated)

    return dilated_image

def pad_image_with_multiplier(
    image: Image.Image, 
    multiplier: int, 
    color: Union[int, tuple[int, int, int]] = 0
) -> Image.Image:
    """Pad image with multiplier.

    Args:
        image (Image.Image): Image to pad.
        multiplier (int): Multiplier.
        fill_color (Union[int, tuple[int, int, int]], optional): Fill color. Defaults to 0.

    Returns:
        Image.Image: Padded image.
    """    
    if isinstance(color, int):
        color = (color, color, color)

    width, height = image.size
    new_width = (width + multiplier - 1) // multiplier * multiplier
    new_height = (height + multiplier - 1) // multiplier * multiplier

    pad_image = Image.new("RGB", (new_width, new_height), color)
    pad_image.paste(image, (0, 0, width, height))

    return pad_image

def color_image_with_mask(image: Image, mask: Image, invert_mask: bool=False):
    """Merge image with mask

    Args:
        image (Image): image
        mask (Image): mask

    Returns:
        Image: merged image
    """    
    mask = mask.convert("L")

    if invert_mask:
        mask = ImageChops.invert(mask)

    alpha_mask = mask.point(lambda x: x / 8)
    
    red_mask = Image.new("RGB", mask.size, (255, 0, 0))

    return Image.composite(red_mask, image, alpha_mask)

def set_seed(seed: int=404):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_mask(im_shape, ratio=1, mask_full_image=False, min_size=0):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(min_size, int(im_shape[0] * ratio)), random.randint(min_size, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 2)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    elif draw_type == 1:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.line(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
            width=random.randint(5, 25),
        )
        
    return mask

def vae_encode_scaling(
    latent: torch.Tensor,
    vae: AutoencoderKL,
    model_type: str  
):
    if model_type == "sd3":
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
    elif model_type == "sdxl" or model_type == "sd":
        latent = latent * vae.config.scaling_factor
    else:
        raise ValueError(f"Not supported vae type {model_type}")
    
    return latent

def vae_decode_scaling(
    latent: torch.Tensor,
    vae: AutoencoderKL,
    model_type: str 
):
    if model_type == "sd3":
        latent = (latent / vae.config.scaling_factor) + vae.config.shift_factor
    elif model_type == "sdxl":
        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
        has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(latent.device, latent.dtype)
            )
            latents_std = (
                torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(latent.device, latent.dtype)
            )
            latent = latent * latents_std / vae.config.scaling_factor + latents_mean
        else:
            latent = latent / vae.config.scaling_factor
    elif model_type == "sd":
        latent = latent / vae.config.scaling_factor
    else:
        raise ValueError(f"Not supported vae type {model_type}")
    
    return latent

def estimate_flow_stage_timesteps(
    num_inference_steps: int,
    stage_count: int,
    start_sigma: float,
    shorten_intermediate_steps: bool,
):
    """
    This function calculates the total number of timesteps needed for a multi-stage inference process.
    It is not a accurate estimation.
    Args:
        num_inference_steps (int): The base number of inference steps for the process.
        stage_count (int): The number of stages in the multi-stage inference.
        start_sigma (float): A ratio (0.0 to 1.0) determining where to start the inference
                           within each stage's timestep range.
        shorten_intermediate_steps (bool): If True, uses progressively fewer steps for 
                                         intermediate stages by halving steps at each stage.
                                         If False, uses the same number of steps for all stages.
    Returns:
        int: The total number of timesteps required for the complete multi-stage inference.
    Note:
        When shorten_intermediate_steps is True, each stage uses half the inference steps
        of the previous stage (scaled by powers of 2). When False, all stages use the
        same base number of inference steps with only the starting point adjusted by start_sigma.
    """
    total_steps = 0

    if not shorten_intermediate_steps:
        init_timestep = min(int(num_inference_steps * start_sigma), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        total_steps = int((num_inference_steps - t_start) * (stage_count) + num_inference_steps)
    else:
        total_steps += num_inference_steps // (2 ** stage_count)
        for stage in range(stage_count - 1, -1, -1):
            scaling_factor = 2 ** stage
            staged_inference_steps = num_inference_steps // scaling_factor
            init_timestep = min(int(staged_inference_steps * start_sigma), staged_inference_steps)
            t_start = max(staged_inference_steps - init_timestep, 0)
            total_steps += staged_inference_steps - t_start
    
    return total_steps

def estimate_diffusion_stage_timesteps(
    num_inference_steps: int,
    stage_count: int,
    start_sigma: float,
    shorten_intermediate_steps: bool,
    base_width: int,
    base_height: int,
    width: int,
    height: int,
    vae_scale_factor: int = 8
):
    """
    This function calculates the total number of timesteps needed for a multi-stage inference process.
    It is not a accurate estimation.
    Args:
        num_inference_steps (int): The base number of inference steps for the process.
        stage_count (int): The number of stages in the multi-stage inference.
        start_sigma (float): A ratio (0.0 to 1.0) determining where to start the inference
                           within each stage's timestep range.
        shorten_intermediate_steps (bool): If True, uses progressively fewer steps for 
                                         intermediate stages by halving steps at each stage.
                                         If False, uses the same number of steps for all stages.
    Returns:
        int: The total number of timesteps required for the complete multi-stage inference.
    Note:
        When shorten_intermediate_steps is True, each stage uses half the inference steps
        of the previous stage (scaled by powers of 2). When False, all stages use the
        same base number of inference steps with only the starting point adjusted by start_sigma.
    """
    total_steps = 0
    
    if not shorten_intermediate_steps:
        init_timestep = min(int(num_inference_steps * start_sigma), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        total_steps = int((num_inference_steps - t_start) * (stage_count) + num_inference_steps)
    else:
        total_steps += num_inference_steps // (2 ** stage_count)
        for stage in range(stage_count - 1, -1, -1):
            scaling_factor = 2 ** stage
            scaled_height = vae_scale_factor * (int(height / scaling_factor) // (vae_scale_factor))
            scaled_width = vae_scale_factor * (int(width / scaling_factor) // (vae_scale_factor))
            
            step_shorten_factor = math.log2(max(2, (base_width / scaled_width) * (base_height / scaled_height)))
            staged_inference_steps = int(num_inference_steps // step_shorten_factor) \
                if shorten_intermediate_steps else num_inference_steps
            init_timestep = min(int(staged_inference_steps * start_sigma), staged_inference_steps)
            t_start = max(staged_inference_steps - init_timestep, 0)
            total_steps += staged_inference_steps - t_start

    return total_steps