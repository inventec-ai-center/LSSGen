# Modified from https://github.com/djghosh13/geneval/blob/main/generation/diffusers_generate.py

import argparse
import json
import time

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm, trange
from diffusers.training_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--model_2",
        type=str,
        default=None,
        help="Huggingface model name, only needed for LCM SDXL."
    )
    parser.add_argument(
        "--upscale_model",
        type=str,
        default=None,
        help="Huggingface model name"
    )
    parser.add_argument(
        "--pipeline_type",
        type=str,
        required=True,
        choices=["flux", "lssflux", "sd", "lsssd", "sdxl", "lsssdxl",
                 "lcmsdxl", "lsslcmsdxl", "sd3", "lsssd3", "cogview4", "lsscogview4"],
        help="pipeline_type name"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=256,
        help="minimal image size in inital stage of LSSGen",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.75,
        help="initial denoising sigma",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--short_intermed",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to run on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="device to run on"
    )
    opt = parser.parse_args()
    return opt

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def main(opt):
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    output_dir = Path(opt.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(opt), f)

    if opt.dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {opt.dtype}. Supported dtypes are: {list(DTYPE_MAP.keys())}")
    
    device = torch.device(opt.device)
    dtype = DTYPE_MAP.get(opt.dtype, torch.bfloat16)

    pipeline_type = opt.pipeline_type.lower()
    if opt.upscale_model is not None and "lss" in pipeline_type:
        from network.models.upsampler import LatentUpSampler
        upsampler = LatentUpSampler.from_pretrained(opt.upscale_model, torch_dtype=dtype).to(device)

    if pipeline_type == "flux":
        from diffusers import FluxPipeline
        pipeline = FluxPipeline.from_pretrained(opt.model, torch_dtype=dtype)
    elif pipeline_type == "lssflux":
        from network.pipelines import LSSFluxPipeline
        pipeline = LSSFluxPipeline.from_pretrained(opt.model, latent_upsampler=upsampler, torch_dtype=dtype).to(device)
    elif pipeline_type == "sd":
        from diffusers import StableDiffusionPipeline
        pipeline = StableDiffusionPipeline.from_pretrained(opt.model, torch_dtype=dtype)
    elif pipeline_type == "lsssd":
        from network.pipelines import LSSStableDiffusionPipeline
        pipeline = LSSStableDiffusionPipeline.from_pretrained(
            opt.model, latent_upsampler=upsampler, torch_dtype=dtype
        ).to(device)
    elif pipeline_type == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(opt.model, torch_dtype=dtype).to(device)
    elif pipeline_type == "lsssdxl":
        from network.pipelines import LSSStableDiffusionXLPipeline
        pipeline = LSSStableDiffusionXLPipeline.from_pretrained(
            opt.model, latent_upsampler=upsampler, torch_dtype=dtype
        ).to(device)
    elif pipeline_type == "lcmsdxl":
        from diffusers import StableDiffusionXLPipeline, LCMScheduler, UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(opt.model_2, torch_dtype=dtype)
        pipeline = StableDiffusionXLPipeline.from_pretrained(opt.model, unet=unet, torch_dtype=dtype)
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    elif pipeline_type == "lsslcmsdxl":
        from diffusers import LCMScheduler, UNet2DConditionModel
        from network.pipelines import LSSStableDiffusionXLPipeline
        unet = UNet2DConditionModel.from_pretrained(opt.model_2, torch_dtype=dtype)
        pipeline = LSSStableDiffusionXLPipeline.from_pretrained(
            opt.model, unet=unet, latent_upsampler=upsampler, torch_dtype=dtype
        ).to(device)
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    elif pipeline_type == "sd3":
        from diffusers import StableDiffusion3Pipeline
        pipeline = StableDiffusion3Pipeline.from_pretrained(opt.model, torch_dtype=dtype).to(device)
    elif pipeline_type == "lsssd3":
        from network.pipelines import LSSStableDiffusion3Pipeline
        pipeline = LSSStableDiffusion3Pipeline.from_pretrained(
            opt.model, latent_upsampler=upsampler, torch_dtype=dtype
        ).to(device)
    elif pipeline_type == "cogview4":
        from diffusers import CogView4Pipeline
        pipeline = CogView4Pipeline.from_pretrained(opt.model, torch_dtype=dtype).to(device)
    elif pipeline_type == "dynacogview4":
        from network.pipelines import LSSCogView4Pipeline
        pipeline = LSSCogView4Pipeline.from_pretrained(
            opt.model, latent_upsampler=upsampler, torch_dtype=dtype
        ).to(device)

    print(opt)
    
    generator = torch.Generator(device=device)
    if "lss" in pipeline_type:
        print("Use custom Latent Space Scaling pipeline.")
    else:
        print("Use normal t2i pipeline.")

    for index, metadata in enumerate(metadatas):
        total_time = 0
        set_seed(opt.seed)
        generator.manual_seed(opt.seed)

        outpath = output_dir / f"{index:0>5}"
        outpath.mkdir(parents=True, exist_ok=True)

        prompt = metadata['prompt']
        batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = outpath / "samples"
        sample_path.mkdir(parents=True, exist_ok=True)
        with open(outpath / "metadata.jsonl", "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0


        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                # Generate images
                start_time = time.time()
                if "lss" in pipeline_type:
                    samples = pipeline(
                        prompt,
                        height=opt.H,
                        width=opt.W,
                        num_inference_steps=opt.steps,
                        guidance_scale=opt.scale,
                        start_sigma=opt.sigma,
                        num_images_per_prompt=min(batch_size, opt.n_samples - sample_count),
                        negative_prompt=opt.negative_prompt or None,
                        generator=generator,
                        min_resolution=opt.min_size,
                        shorten_intermediate_steps=opt.short_intermed,
                    ).images
                else:
                    samples = pipeline(
                        prompt,
                        height=opt.H,
                        width=opt.W,
                        num_inference_steps=opt.steps,
                        guidance_scale=opt.scale,
                        num_images_per_prompt=min(batch_size, opt.n_samples - sample_count),
                        negative_prompt=opt.negative_prompt or None,
                        generator=generator,
                    ).images
                end_time = time.time()
                total_time += end_time - start_time
                for sample in samples:
                    sample.save(sample_path / f"{sample_count:05}.png")
                    sample_count += 1
        del all_samples

        with open(outpath / "result.json", "w") as fp:
            json.dump({"time": total_time, "n_sample": opt.n_samples, "batch_size": opt.batch_size}, fp)
    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
