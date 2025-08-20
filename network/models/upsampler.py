# TODO: LICENSE

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers import ModelMixin, AutoencoderKL
from diffusers.utils import BaseOutput
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from diffusers.models.unets.unet_2d_blocks import (
    AutoencoderTinyBlock,
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)
from diffusers.models.upsampling import Upsample2D
from network.utils import vae_decode_scaling, vae_encode_scaling

class LatentUpSampler(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (64,),
        is_up_samples: Tuple[bool, ...] = [True],
        strides: Tuple[int, ...] = (2,),
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        interpolate: str = "nearest", # "nearest", "bilinear", "bicubic"
    ):
        super().__init__()
        
        self.upsampler = ResNetUpsampler(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            strides=strides,
            is_up_samples=is_up_samples,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            interpolate=interpolate
        )

    @staticmethod
    def vae_encode_image(
        image: Image.Image | torch.Tensor, 
        vae: AutoencoderKL,
        model_type: str,
        sample_mode: str = "sample",
        generator: torch.Generator = None
    ):
        img2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
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
    
    @staticmethod
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
            return transforms.ToPILImage()(image.squeeze())
        elif output_type.lower() == "numpy":
            return image.detach().cpu().permute(0, 2, 3, 1).float().numpy()

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
        """
        sample = self.upsampler(latent)

        return sample
    
class ResNetUpsampler(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (64,),
        strides: Tuple[int, ...] = (2,),
        is_up_samples: Tuple[bool] = [True],
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        interpolate: str = "nearest", # "nearest", "bilinear", "bicubic"
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        # up
        prev_output_channel = block_out_channels[0]
        for i, block_out_channel in enumerate(block_out_channels):
            up_block = ResNetBlockTranspose2d(
                in_channels=prev_output_channel,
                out_channels=block_out_channel,
                act_fn=act_fn,
                num_groups=norm_num_groups,
                interpolate=interpolate,
                is_upsample=is_up_samples[i],
                stride=strides[i],
            )
            self.up_blocks.append(up_block)
            prev_output_channel = block_out_channel

        # out
        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1])
        self.act_out = get_activation(act_fn)
        self.conv_out = nn.Conv2d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        sample: torch.Tensor,    
    ):
        """_summary_

        Args:
            sample (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """    
        sample = self.conv_in(sample)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        sample = self.norm_out(sample)
        sample = self.act_out(sample)
        sample = self.conv_out(sample)

        return sample

class ResNetBlockTranspose2d(nn.Module):
    r"""ResNet Block Transpose 2D"""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        padding: int = 1,
        act_fn: str = "silu",
        num_groups: int = 32,
        interpolate: str = "nearest", # "nearest", "bilinear", "bicubic"
        eps: float = 1e-6,
        is_upsample: bool = True,
        stride: int = 2,
    ):
        super().__init__()

        self.interpolate = interpolate
        self.is_upsample = is_upsample
        self.stride = stride

        self.norm1 = nn.GroupNorm(num_groups, in_channels, eps=eps)
        if stride > 1 and is_upsample:
            self.conv1 = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=padding,
            )
        elif stride > 1:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )

        self.norm2 = nn.GroupNorm(num_groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
        )

        self.act = get_activation(act_fn)

        # Skip connection projection
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_proj = nn.Identity()


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Computes the output of the residual block

        Args:
            x (torch.Tensor): A 4D Torch Tensor which is the input to the Transposed Residual Block.

        Returns:
            4D Torch Tensor after applying the desired functions as specified while creating the
            object.
        """
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.act(hidden_states)

        # Skip connection
        if self.stride != 1:
            # Upsample skip connection if stride > 1
            skip = F.interpolate(
                self.skip_proj(input_tensor), 
                scale_factor=self.stride if self.is_upsample else 1 / self.stride, 
                mode=self.interpolate
            )
        else:
            # Direct skip connection if stride = 1
            skip = self.skip_proj(input_tensor)

        return hidden_states + skip