from contextlib import contextmanager
from functools import wraps
import gc
import os
from pathlib import Path
import time
from typing import Any

from diffusers import DiffusionPipeline
from PIL import Image
from pruna import SmashConfig, smash
import torch
from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
from torchao.quantization import quantize_
from torch.profiler import record_function

from models.sdxl_base import Timings
from output import save_with_prompt

# pruna's default is ~/.cache/pruna, which ignores XDG_CACHE_HOME and would land on the NFS home.
PRUNA_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pruna"

# Selective quantization after https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/:
# skip layers where the weight or activation is too small for FP4 matmuls to beat the activation-quantization
# overhead (min dimension < 1024), and embedding-like layers. In the SDXL U-Net that leaves the linears of the
# 1280-channel transformer blocks (1024 tokens at 1024x1024).
MIN_FEATURES = 1024
SKIPPED_NAME_PARTS = (
    "attn2.to_k", "attn2.to_v",  # cross-attention keys/values see only the 77 text tokens
    "time_emb_proj",  # one timestep embedding per sample
    # proj_in gets a non-contiguous (channels-last reshaped) input, for which F.linear needs aten.expand on the
    # weight, which torchao 0.16's NVFP4Tensor does not implement (fails in eager and under torch.compile).
    "proj_in",
    "time_embedding.", "add_embedding.",  # embeddings
)


def nvfp4_filter(module: torch.nn.Module, name: str) -> bool:
    return (
        isinstance(module, torch.nn.Linear)
        and min(module.in_features, module.out_features) >= MIN_FEATURES
        and module.in_features % 16 == 0 and module.out_features % 16 == 0
        and not any(part in name for part in SKIPPED_NAME_PARTS)
    )


def cast_io_to_bf16(module: torch.nn.Linear):
    """Run an NVFP4 linear on bf16 activations and return the caller's dtype. With fp16 inputs, torchao 0.16's
    NVFP4 matmul returns inf; with bf16 inputs it is correct."""
    forward = module.forward

    def bf16_forward(x: torch.Tensor) -> torch.Tensor:
        return forward(x.to(torch.bfloat16)).to(x.dtype)

    module.forward = bf16_forward


class SDXLPrunaNVFP4:
    """SDXL base 1.0 in fp16 with NVFP4 (torchao) on selected U-Net linears, then smashed with pruna, with
    per-component timing. Needs a Blackwell GPU (compute capability >= 10.0).

    torchao 0.16 only quantizes bf16/fp32 weights to NVFP4 and only computes correctly with bf16 activations, so the
    selected linears are cast to bf16 before quantizing and wrapped to cast their input to bf16 and their output
    back to fp16. The rest of the pipeline stays fp16, like the reference."""

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, smash_config: list[str] | dict[str, Any] | None = None, device: str = "cuda"):
        """smash_config is passed to pruna's SmashConfig, e.g. ["deepcache", "torch_compile"] or
        {"deepcache": True, "deepcache_interval": 3}. Defaults to DeepCache interval 3 + torch.compile."""
        self.smash_config = (
            {"deepcache": True, "deepcache_interval": 3, "torch_compile": True} if smash_config is None else smash_config
        )
        self.device = device
        self.pipe = None
        # Per-label durations of the generation being timed; None while not timing.
        self._timings: dict[str, list[float]] | None = None
        # pruna's full configuration including defaults, known after prepare().
        self._resolved_smash_config: str | None = None
        self._quantized_layers: int | None = None

    def describe(self) -> dict:
        """What exactly this model is, for run records."""
        return {
            "class": type(self).__name__,
            "model_id": self.model_id,
            "dtype": "float16",
            "smash_config": self.smash_config,
            "smash_config_resolved": self._resolved_smash_config,
            "nvfp4": {
                "config": "NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=True)",
                "scope": f"unet nn.Linear, min(in, out) >= {MIN_FEATURES}, skipping {', '.join(SKIPPED_NAME_PARTS)}",
                "weights_quantized_from": "bfloat16",
                "activations": "cast fp16 -> bf16 before each quantized linear, output cast back to fp16",
                "quantized_layers": self._quantized_layers,
            },
        }

    def prepare(self):
        """Load the pipeline onto the device, quantize U-Net linears to NVFP4, smash it, then instrument its components."""
        pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.to(self.device)

        targets = [name for name, module in pipe.unet.named_modules() if nvfp4_filter(module, name)]
        for name in targets:
            pipe.unet.get_submodule(name).to(torch.bfloat16)
        quantize_(pipe.unet, NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=True),
                  filter_fn=nvfp4_filter)
        for name in targets:
            cast_io_to_bf16(pipe.unet.get_submodule(name))
        self._quantized_layers = len(targets)
        print(f"NVFP4: quantized {len(targets)} of {sum(isinstance(m, torch.nn.Linear) for m in pipe.unet.modules())} U-Net linears")

        config = SmashConfig(self.smash_config, device=self.device, cache_dir_prefix=PRUNA_CACHE_DIR)
        config.disable_saving()  # inference only; don't keep copies of the model around for saving
        self.pipe = smash(model=pipe, smash_config=config)
        self._resolved_smash_config = str(config)
        print(f"smashed with {config}")

        # Instrument after smashing: pruna replaces unet.forward (torch.compile, DeepCache), and the
        # timing wrappers must sit outside the compiled region to avoid graph breaks.
        self._instrument(self.pipe.text_encoder, "forward", "text_encoder")
        self._instrument(self.pipe.text_encoder_2, "forward", "text_encoder_2")
        self._instrument(self.pipe.unet, "forward", "unet_step")
        self._instrument(self.pipe.vae, "decode", "vae_decode")

    def warm_up(self, prompt: str = "A warm cache", save: bool = True):
        """Run one untimed generation: compilation, CUDA context, cuDNN autotuning, allocator caches."""
        image = self.pipe(prompt=prompt).images[0]
        if save:
            save_with_prompt(image, prompt)

    def unload(self):
        """Free the pipeline's GPU memory, e.g. before loading another model."""
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def generate_timed(self, prompt: str, seed: int | None = None, save: bool = True) -> tuple[Image.Image, Timings]:
        """Generate an image, save it with its prompt into outputs/ if save, and return it with how long each component took."""
        generator = None if seed is None else torch.Generator(self.device).manual_seed(seed)
        self._timings = {"text_encoder": [], "text_encoder_2": [], "unet_step": [], "vae_decode": []}
        try:
            with self._timed("total"):
                image = self.pipe(prompt=prompt, generator=generator).images[0]
            t = self._timings
        finally:
            self._timings = None

        if save:
            save_with_prompt(image, prompt)
        return image, Timings(
            text_encoder_ms=sum(t["text_encoder"]),
            text_encoder_2_ms=sum(t["text_encoder_2"]),
            unet_step_ms=t["unet_step"],
            vae_decode_ms=sum(t["vae_decode"]),
            total_ms=t["total"][0],
        )

    @contextmanager
    def _timed(self, label: str):
        # Synchronizing before and after makes the duration cover the component's CUDA work,
        # not just its kernel launches. record_function labels the range for torch.profiler.
        if self._timings is None:
            yield
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        with record_function(label):
            yield
            torch.cuda.synchronize()
        self._timings.setdefault(label, []).append((time.perf_counter() - start) * 1000)

    def _instrument(self, obj, method: str, label: str):
        original = getattr(obj, method)

        @wraps(original)
        def wrapper(*args, **kwargs):
            with self._timed(label):
                return original(*args, **kwargs)

        setattr(obj, method, wrapper)
