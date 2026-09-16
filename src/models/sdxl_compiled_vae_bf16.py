from contextlib import contextmanager
from functools import wraps
import gc
import time

from diffusers import DiffusionPipeline
from PIL import Image
import torch
from torch.profiler import record_function

from models.sdxl_base import Timings
from models.sdxl_base_vae_bf16 import cast_vae_decode
from output import save_with_prompt


class SDXLCompiledVAEBF16:
    """SDXL base 1.0 in fp16 with torch.compile on the U-Net, the VAE decoder in bfloat16, and torch.compile on
    vae.decode, with per-component timing. 50 steps, guidance and the invisible watermark as in the base model.

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
        # Per-label durations of the generation being timed; None while not timing.
        self._timings: dict[str, list[float]] | None = None

    def describe(self) -> dict:
        """What exactly this model is, for run records."""
        return {"class": type(self).__name__, "model_id": self.model_id, "dtype": "float16", "vae_decode_dtype": "bfloat16",
                "unet_compile": f"torch.compile(unet.forward, mode={self.compile_mode!r})",
                "vae_decode_compile": f"torch.compile(vae.decode, mode={self.vae_compile_mode!r})", "watermark": True}

    def prepare(self):
        """Load the pipeline onto the device and instrument its components."""
        self.pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.to(self.device)
        self.pipe.vae.to(torch.bfloat16)
        cast_vae_decode(self.pipe.vae, torch.bfloat16)
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
