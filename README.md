# Leveraging Latent Space Scaling in Flow and Diffusion for Efficient Text to Image Generation
This repository contains the code and resources for the paper "Leveraging Latent Space Scaling in Flow and Diffusion for Efficient Text to Image Generation".  
Accepted at ICCV AIGENS workshop 2025.  

[![arXiv](https://img.shields.io/badge/arXiv-2507.16154-b31b1b.svg)](https://arxiv.org/abs/2507.16154) [![alphaXiv](https://img.shields.io/badge/alphaXiv-2507.16154-009639.svg)](https://www.alphaxiv.org/abs/2507.16154) [![CVF](https://img.shields.io/badge/CVF-Open%20Access-orange.svg)](#)

## Abstract
TL;DR: This framework accelerates text-to-image generation by shifting the early timesteps of diffusion and flow to a lower resolution in latent space. Compared to traditional methods, it achieves better image quality.  
> Flow matching and diffusion models have shown impressive results in text-to-image generation, producing photorealistic images through an iterative denoising process. A common strategy to speed up synthesis is to perform early denoising at lower resolutions. However, traditional methods that downscale and upscale in pixel space often introduce artifacts and distortions. These issues arise when the upscaled images are re-encoded into the latent space, leading to degraded final image quality.
To address this, we propose **Latent Space Scaling Generation (LSSGen)**, a framework that performs resolution scaling directly in the latent space using a lightweight latent upsampler. Without altering the Transformer or U-Net architecture, LSSGen improves both efficiency and visual quality while supporting flexible multi-resolution generation. Our comprehensive evaluation covering text-image alignment and perceptual quality shows that LSSGen significantly outperforms conventional scaling approaches. When generating $1024^2$ images at similar speeds, it achieves up to 246\% TOPIQ score improvement.

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Diffusers

## Usage
### Diffusers style pipelines
To use the LSSGen framework with Diffusers pipeline style, you can follow the example below.  

#### Latent Upsampler Preparation
```python
dtype = torch.bfloat16
device = "cuda"
from network.models.upsampler import LatentUpSampler
upsampler = LatentUpSampler.from_pretrained(UPSAMPLER_PATH, torch_dtype=dtype).to(device)
```

Latent Upsampler model urls:
- FLUX
- Stable Diffusion 3
- Stable Diffusion XL
- Stable Diffusion
- CogView4

#### FLUX pipeline:
The FLUX pipeline supports all FLUX.1 text to image models, including FLUX.1-dev, FLUX.1-schnell, and FLUX.1-Krea-dev.  
```python
from network.pipelines import LSSFluxPipeline
pipeline = LSSFluxPipeline.from_pretrained(MODEL_PATH, latent_upsampler=upsampler, torch_dtype=dtype).to(device)

image = pipeline(
    prompt, height=1024, width=1024, num_inference_steps=50, guidance_scale=3.5, 
    start_sigma=0.75, shorten_intermediate_steps=True
).images[0]
```

#### Stable Diffusion 3 pipeline:
The Stable Diffusion 3 pipeline supports all Stable Diffusion 3 models, including SD3, SD3.5.  
```python
from network.pipelines import LSSStableDiffusion3Pipeline
pipeline = LSSStableDiffusion3Pipeline.from_pretrained(MODEL_PATH, latent_upsampler=upsampler, torch_dtype=dtype).to(device)

image = pipeline(
    prompt, height=1024, width=1024, num_inference_steps=40, guidance_scale=4.5, 
    start_sigma=0.75, shorten_intermediate_steps=True
).images[0]
```

#### Stable Diffusion XL pipeline:
The Stable Diffusion XL pipeline supports all Stable Diffusion XL model variants, including SDXL, and Playground v2.5.
```python
from network.pipelines import LSSStableDiffusionXLPipeline
pipeline = LSSStableDiffusionXLPipeline.from_pretrained(MODEL_PATH, latent_upsampler=upsampler, torch_dtype=dtype).to(device)

image = pipeline(
    prompt, height=1024, width=1024, num_inference_steps=50, guidance_scale=5, 
    start_sigma=0.75, shorten_intermediate_steps=False
).images[0]
```

#### Stable Diffusion pipeline:
```python
from network.pipelines import LSSStableDiffusionPipeline
pipeline = LSSStableDiffusionPipeline.from_pretrained(MODEL_PATH, latent_upsampler=upsampler, torch_dtype=dtype).to(device)

image = pipeline(
    prompt, height=1024, width=1024, num_inference_steps=50, guidance_scale=7.5, 
    start_sigma=0.75, shorten_intermediate_steps=False
).images[0]
```

#### CogView4 pipeline:
```python
from network.pipelines import LSSCogView4Pipeline
pipeline = LSSCogView4Pipeline.from_pretrained(MODEL_PATH, latent_upsampler=upsampler, torch_dtype=dtype).to(device)

image = pipeline(
    prompt, height=1024, width=1024, num_inference_steps=50, guidance_scale=3.5, 
    start_sigma=0.75, shorten_intermediate_steps=True
).images[0]
```

### Or you can use the GenEval inference script to batch generate images with LSSGen.
```bash
python inference_dyn_scaling_t2i.py <GENEVAL_PROMPTS_JSONL> \
    --model <MODEL_PATH> \
    --upscale_model <UPSAMPLER_PATH> \
    --pipeline_type <PIPELINE_TYPE> \ # e.g., flux, lssflux, sd, lsssd etc.
    --output_dir <OUTPUT_DIR> \
    --num_inference_steps <NUM_STEPS> \
    --guidance_scale <GUIDANCE_SCALE> \
    --start_sigma 0.75 \
    --shorten_intermediate_steps True \
    --min_size 512 # minimal image size in initial stage of LSSGen
```

## Parameters
- `Shorten Intermediate Steps`: If set to `True`, the LSSGen will shorten the intermediate steps when resolution is lower than the standard resolution. This is useful for speeding up the generation process while maintaining quality in flow based models.  
- `Start Sigma`: The initial sigma value for the denoising process. A value for controling the trade-off between speed and quality. 0.75 is optimal for best quality, but you can try other values like 0.67 for faster generation.  

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.