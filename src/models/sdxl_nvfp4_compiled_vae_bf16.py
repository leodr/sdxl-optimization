from contextlib import contextmanager
from functools import wraps
import gc
import time

from diffusers import DiffusionPipeline
from PIL import Image
import torch
from torch.profiler import record_function
from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
from torchao.quantization import quantize_

from models.sdxl_base import Timings
from models.sdxl_base_vae_bf16 import cast_vae_decode
from models.sdxl_pruna_nvfp4 import MIN_FEATURES, SKIPPED_NAME_PARTS, cast_io_to_bf16, nvfp4_filter
from output import save_with_prompt


class SDXLNVFP4CompiledVAEBF16:
    """SDXL base 1.0 in fp16 with NVFP4 on selected U-Net linears, torch.compile on the U-Net, the VAE decoder in
    bfloat16, and torch.compile on vae.decode, with per-component timing. 50 steps, guidance and the invisible
    watermark as in the base model. Identical to SDXLCompiledVAEBF16 except for NVFP4, which uses the same layer
    selection and bf16 activation casts as SDXLPrunaNVFP4.

    The SDXL VAE overflows in fp16, so the pipeline upcasts it to float32 for every decode. bf16 has float32's range
    and fast kernels on Blackwell. With a bf16 VAE the pipeline no longer upcasts and passes the fp16 latents
    through, so decode casts its input to bf16 and its output back to float32; the watermark and postprocessing
    then see the same dtype as with the float32 decoder."""

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, compile_mode: str = "max-autotune-no-cudagraphs", vae_compile_mode: str = "max-autotune-no-cudagraphs",
                 device: str = "cuda"):
        self.compile_mode = compile_mode
        self.vae_compile_mode = vae_compile_mode
        self.device = device
        self.pipe = None
        self._quantized_layers: int | None = None
        # Per-label durations of the generation being timed; None while not timing.
        self._timings: dict[str, list[float]] | None = None

    def describe(self) -> dict:
        """What exactly this model is, for run records."""
        return {"class": type(self).__name__, "model_id": self.model_id, "dtype": "float16", "vae_decode_dtype": "bfloat16",
                "unet_compile": f"torch.compile(unet.forward, mode={self.compile_mode!r})",
                "vae_decode_compile": f"torch.compile(vae.decode, mode={self.vae_compile_mode!r})", "watermark": True,
                "nvfp4": {
                    "config": "NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=True)",
                    "scope": f"unet nn.Linear, min(in, out) >= {MIN_FEATURES}, skipping {', '.join(SKIPPED_NAME_PARTS)}",
                    "weights_quantized_from": "bfloat16",
                    "activations": "cast fp16 -> bf16 before each quantized linear, output cast back to fp16",
                    "quantized_layers": self._quantized_layers,
                }}

    def prepare(self):
        """Load the pipeline onto the device and instrument its components."""
        self.pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.to(self.device)
        self.pipe.vae.to(torch.bfloat16)
        cast_vae_decode(self.pipe.vae, torch.bfloat16)

        unet = self.pipe.unet
        targets = [name for name, module in unet.named_modules() if nvfp4_filter(module, name)]
        for name in targets:
            unet.get_submodule(name).to(torch.bfloat16)
        quantize_(unet, NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=True),
                  filter_fn=nvfp4_filter)
        for name in targets:
            cast_io_to_bf16(unet.get_submodule(name))
        self._quantized_layers = len(targets)
        print(f"NVFP4: quantized {len(targets)} of {sum(isinstance(m, torch.nn.Linear) for m in unet.modules())} U-Net linears")

        # Same as pruna's torch_compile without DeepCache: compile unet.forward as a whole. The cast wrapper is
        # compiled together with the decode. The timing wrappers below stay outside the compiled regions.
        self.pipe.unet.forward = torch.compile(self.pipe.unet.forward, mode=self.compile_mode)
        self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode=self.vae_compile_mode)

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
