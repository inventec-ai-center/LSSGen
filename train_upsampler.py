from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from dataclasses import dataclass

from diffusers.optimization import (
    get_cosine_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_scheduler
)
import itertools
from numerize.numerize import numerize
from pathlib import Path
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
# from lpips import LPIPS
from tqdm.auto import tqdm

import argparse
import copy
import diffusers
import logging
import random
import math
import numpy as np
import torch
import transformers

from diffusers import AutoencoderKL

from network.datasets.DIV2K import DIV2KScalingDataset
from network.models.upsampler import LatentUpSampler

logger = get_logger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Train a UNet model for image inpainting")
    
    # Training parameters
    parser.add_argument("--image_size", type=int, nargs='+', required=True, help="the generated image resolution")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16, help="how many images to sample during evaluation")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_image_epochs", type=int, default=5)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose"
                            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
                            "and an Nvidia Ampere GPU."
                        ))
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between '
                            '["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
    )
    parser.add_argument("--scale_lr", action="store_true", help="Scale the learning rate by the number of GPUs.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs.")
    parser.add_argument("--perceptual_loss_scale", 
                        type=float, 
                        default=1.0,
                        help="Perceptual loss scale, which is implented with LPIPS."
    )
    parser.add_argument("--latent_loss_scale", 
                        type=float,  
                        default=1.0,
                        help="Latent space loss scale, which is implented with MSE."
    )
    parser.add_argument("--dataset", type=str, default="div2k", choices=["div2k", "coco"],
                        help=(
                            'The scheduler type to use. Choose between '
                            '["div2k", "coco"]'
                        ),
    )
    parser.add_argument("--model_type", type=str, default="sd3", choices=["sd3", "sdxl", "sd"],
                        help=(
                            'The scheduler type to use. Choose between '
                            '["sd3", "sdxl", "sd"]'
                        ),
    )
    parser.add_argument("--target_type", type=str, default="origin", choices=["origin", "nearest"],
                        help=(
                            'The scheduler type to use. Choose between '
                            '["origin", "nearest"]'
                        ),
    )

    # Optimizer
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    # Checkpoints
    parser.add_argument("--checkpoints_total_limit", type=int, default=5, help="maximum number of checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, help="path to checkpoint to resume from")
    parser.add_argument("--checkpointing_epochs", type=int, default=5,
                        help=(
                            "Save a checkpoint of the training state every X updates. These checkpoints can be used "
                            "both as final checkpoints in case they are better than the last checkpoint and are "
                            "suitable for resuming training using `--resume_from_checkpoint`."
                        ))

    # Paths
    parser.add_argument("--data_dir", type=str, default="/home/jovyan/aic-nas2/datasets/DIV2K/", help="directory containing the super resolution dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="model name locally and on the HF Hub")
    parser.add_argument("--log_dir", type=str, default="logs", help="directory to save the logs")
    parser.add_argument("--pretrained_model_path", type=str, default="/home/jovyan/aic-nas2/Diffusion/ImageSynthesis/FLUX.1-dev", help="path to pretrained model")
    parser.add_argument("--model_path", type=str, default=None, help="path to pretrained model")
    parser.add_argument("--test_img_dir", type=str, default="/home/jovyan/aic-nas2/datasets/CelebA-HQ-256/test", help="path to test images")
    parser.add_argument("--test_img_count", type=int, default=10, help="number of images to test")

    # Other settings
    parser.add_argument("--push_to_hub", action="store_true", help="whether to upload model to HF Hub")
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="overwrite old model when re-running")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_run", action="store_true", help="If true, only runs on partial training and validation dataset")

    args = parser.parse_args()
    return args

def main():
    config = get_args()

    img2tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def vae_encode_image(
        image: Image.Image | torch.Tensor, 
        vae: AutoencoderKL,
        model_type: str,
        sample_mode: str = "sample",
        generator: torch.Generator = None
    ):
        if isinstance(image, Image.Image):
            image = img2tensor(image).unsqueeze(0)

        encoder_output = vae.encode(image.to(dtype=vae.dtype, device=vae.device))
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            encoder_output = encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            encoder_output = encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            encoder_output = encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")
        
        encoder_output = vae_encode_scaling(encoder_output, vae, model_type)

        return encoder_output
    
    def vae_decode_latent(
        latent: torch.Tensor,
        vae: AutoencoderKL,
        model_type: str,
        output_type: str = "tensor",
        denormalize: bool = True
    ):
        latent = vae_decode_scaling(latent, vae, model_type)

        image = vae.decode(latent, return_dict=False)[0]

        # Denormalize
        if denormalize:
            image = (image * 0.5 + 0.5).clamp(0, 1)

        # Convert type
        if output_type.lower() == "tensor":
            return image
        elif output_type.lower() == "pil":
            return transforms.ToPILImage(image)
        elif output_type.lower() == "numpy":
            return image.detach().cpu().permute(0, 2, 3, 1).float().numpy()

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

    if len(config.image_size) == 1:
        image_size_w = config.image_size[0]
        image_size_h = config.image_size[0]
        config.image_size = image_size_w
    elif len(config.image_size) == 2:
        image_size_w = config.image_size[0]
        image_size_h = config.image_size[1]
        config.image_size = max(image_size_w, image_size_h)
    else:
        raise ValueError("Image size must be a list of length 1 or 2")

    # Dataset and DataLoader
    train_data_path = "training" if config.dataset == "div2k" else "train2017"
    val_data_path = "validation/images" if config.dataset == "div2k" else "val2017"
    train_dataset = DIV2KScalingDataset(
        image_dir=Path(config.data_dir) / train_data_path, 
        resolution=(image_size_w, image_size_h),
        target_resolution=(image_size_w * 2, image_size_h * 2),
        test_run=config.test_run,
        target_type=config.target_type
    )
    val_dataset = DIV2KScalingDataset(
        image_dir=Path(config.data_dir) / val_data_path, 
        resolution=(image_size_w, image_size_h),
        target_resolution=(image_size_w * 2, image_size_h * 2),
        test_run=config.test_run,
        target_type=config.target_type
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, \
                                  shuffle=True, num_workers=12)
    val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch_size, \
                                shuffle=False, num_workers=12)
    
    num_update_steps_per_epoch = len(train_dataloader)
    if config.max_train_steps is not None:
        max_train_steps = config.max_train_steps
        if config.num_epochs is not None:
            raise ValueError("You cannot provide both max_train_steps and num_epochs")
        num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    elif config.num_epochs is not None:
        num_epochs = config.num_epochs
        max_train_steps = num_update_steps_per_epoch * num_epochs
    else:
        raise ValueError("Either max_train_steps or num_epochs must be provided")

    # Accelerator
    output_dir = Path(config.output_dir)
    project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=Path(config.log_dir) / output_dir.name,
        total_limit=config.checkpoints_total_limit
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard" if not config.test_run else None,
        project_config=project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if config.seed:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")
    if config.model_path is None:
        model = LatentUpSampler.from_config(config.pretrained_model_path, subfolder="latentsampler")
    else:
        model = LatentUpSampler.from_config(config.model_path)
    lpips = LPIPS(net_type="alex", reduction="mean", normalize=False)

    # We only train the sampler
    vae.requires_grad_(False)
    model.requires_grad_(True)
    lpips.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    vae.to(accelerator.device, dtype=weight_dtype)
    lpips.to(accelerator.device)

    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.adam_weight_decay
    )
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=max_train_steps,
    )

    model, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, lr_scheduler)

    accelerator.register_for_checkpointing(lr_scheduler)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("upsample", config=vars(config))

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Num steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Sampler parameters = {numerize(model.num_parameters())}")
    global_step = 0
    first_epoch = 0

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)

    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = Path(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [d for d in Path(config.output_dir).iterdir() if d.name.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.name.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path.as_posix())
            global_step = int(path.name.split("-")[1])

            resume_global_step = global_step * config.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * config.gradient_accumulation_steps)

    # Training Loop
    def training_loop(batch, vae, sampler, lpips, weight_dtype, model_type):
        highres_images = batch["target"]
        lowres_images = batch["image"]

        # Prepare latents
        high_latent = vae.encode(highres_images.to(weight_dtype)).latent_dist.sample()
        low_latent = vae.encode(lowres_images.to(weight_dtype)).latent_dist.sample()
        low_latent = vae_encode_scaling(low_latent, vae, model_type)

        # Sample
        sampled_latent = sampler(low_latent)

        sampled_latent = vae_decode_scaling(sampled_latent, vae, model_type)

        # Pixel Space
        lowres_sampled = vae.decode(sampled_latent.to(weight_dtype)).sample.to(accelerator.device).clamp(-1, 1)

        # Loss
        latent_loss = F.mse_loss(sampled_latent, high_latent)
        perceptual_loss = lpips(highres_images, lowres_sampled)

        return latent_loss, perceptual_loss

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    factor = 1.0 / len(train_dataloader)

    for epoch in range(first_epoch, num_epochs):
        epoch_loss = 0
        epoch_latent_loss = 0
        epoch_perceptual_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if config.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % config.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(model):
                latent_loss, perceptual_loss = training_loop(batch, vae, model, lpips, weight_dtype, config.model_type)
                loss = config.latent_loss_scale * latent_loss + config.perceptual_loss_scale * perceptual_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        model.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "Loss/loss": loss.detach().item(), 
                "Loss/latent_loss": latent_loss.detach().item(),
                "Loss/perceptual_loss": perceptual_loss.detach().item(),
                "General/lr": lr_scheduler.get_last_lr()[0], 
                "General/epoch": epoch
            }
            epoch_loss += logs["Loss/loss"] * factor
            epoch_latent_loss += logs["Loss/latent_loss"] * factor
            epoch_perceptual_loss += logs["Loss/perceptual_loss"] * factor
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                logger.info("*** Training finished ***")
                break

        accelerator.log({
                "Loss/epoch_loss": epoch_loss,
                "Loss/epoch_latent_loss": epoch_latent_loss,
                "Loss/epoch_perceptual_loss": epoch_perceptual_loss,
            }, step=global_step)
        
        if accelerator.sync_gradients:
            if epoch % config.checkpointing_epochs == 0 and epoch > 0:
                if accelerator.is_main_process:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path.as_posix())
                    logger.info(f"Saved state to {save_path.as_posix()}")

        # Validation
        if accelerator.is_main_process:
            model.eval()
            losses = 0
            latent_losses = 0
            perceptual_losses = 0
            factor = 1.0 / len(val_dataloader)
            for step, batch in enumerate(val_dataloader):
                with torch.no_grad():
                    latent_loss, perceptual_loss = training_loop(
                        batch, vae, model, lpips, weight_dtype, config.model_type
                    )

                    loss = config.latent_loss_scale * latent_loss + config.perceptual_loss_scale * perceptual_loss
                    latent_losses += latent_loss.detach().item() * factor
                    perceptual_losses += perceptual_loss.detach().item() * factor
                    losses += loss.detach().item() * factor
                    
            logs = {
                "ValLoss/val_loss": losses, 
                "ValLoss/val_latent_loss": latent_losses,
                "ValLoss/val_perceptual_loss": perceptual_losses,
            }
            accelerator.log(logs, step=global_step)
            
            if (epoch + 1) % config.save_image_epochs == 0:
                # setup datapath
                root_dir = Path(config.test_img_dir)
                imgs_dir = root_dir / "images"

                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(42)
                images = []
                count = 0
                with torch.no_grad():
                    for img_path in imgs_dir.iterdir():
                        if img_path.suffix != ".png" and img_path.suffix != ".jpg":
                            continue
                        if count >= config.test_img_count:
                            continue
                        count += 1

                        high_res = Image.open(img_path).convert("RGB")
                        width, height = high_res.size
                        low_res = high_res.resize((width // 2, height // 2))

                        low_res_latent = vae_encode_image(low_res, vae, generator=generator, model_type=config.model_type)
                        
                        sampled_latent = model(low_res_latent)
                        scaled_image = vae_decode_latent(
                            sampled_latent, vae, output_type="numpy", model_type=config.model_type).squeeze()

                        images.append(scaled_image)

                for tracker in accelerator.trackers:
                    np_images = np.stack([img for img in images])
                    tracker.writer.add_images("Upsample/validation", np_images, epoch, dataformats="NHWC")

                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        unwrap_model = accelerator.unwrap_model(model)
        unwrap_model.save_pretrained(config.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main()