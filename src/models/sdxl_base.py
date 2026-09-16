from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import gc
import time

from diffusers import DiffusionPipeline
from PIL import Image
import torch
from torch.profiler import record_function

from output import save_with_prompt


@dataclass
class Timings:
    """Wall-clock durations of one generation, in ms."""

    text_encoder_ms: float
    text_encoder_2_ms: float
    unet_step_ms: list[float] = field(repr=False)
    vae_decode_ms: float
    total_ms: float

    @property
    def unet_total_ms(self) -> float:
        return sum(self.unet_step_ms)

    @property
    def unet_step_median_ms(self) -> float:
        return sorted(self.unet_step_ms)[len(self.unet_step_ms) // 2]

    @property
    def other_ms(self) -> float:
        """Everything not measured above: tokenization, scheduler, image postprocessing, ..."""
        return self.total_ms - self.text_encoder_ms - self.text_encoder_2_ms - self.unet_total_ms - self.vae_decode_ms

    def format(self) -> str:
        rows = [
            ("text_encoder (CLIP ViT-L)", self.text_encoder_ms),
            ("text_encoder_2 (OpenCLIP ViT-bigG)", self.text_encoder_2_ms),
            (f"unet, all {len(self.unet_step_ms)} steps", self.unet_total_ms),
            ("vae decode", self.vae_decode_ms),
            ("other (scheduler, image postprocessing, ...)", self.other_ms),
        ]
        lines = [f"{'component':<46}{'ms':>10}{'share':>8}"]
        lines += [f"{name:<46}{ms:>10.1f}{ms / self.total_ms:>8.1%}" for name, ms in rows]
        lines.append(f"{'pipeline total':<46}{self.total_ms:>10.1f}")
        steps = self.unet_step_ms
        lines.append(
            f"\nunet single step (ms)\n  first {steps[0]:.1f}  mean {self.unet_total_ms / len(steps):.1f}  "
            f"median {self.unet_step_median_ms:.1f}  min {min(steps):.1f}  max {max(steps):.1f}"
        )
        return "\n".join(lines)


class SDXLBase:
    """SDXL base 1.0 in fp16, with per-component timing."""

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None
        # Per-label durations of the generation being timed; None while not timing.
        self._timings: dict[str, list[float]] | None = None

    def describe(self) -> dict:
        """What exactly this model is, for run records."""
        return {"class": type(self).__name__, "model_id": self.model_id, "dtype": "float16"}

    def prepare(self):
        """Load the pipeline onto the device and instrument its components."""
        self.pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.to(self.device)

        self._instrument(self.pipe.text_encoder, "forward", "text_encoder")
        self._instrument(self.pipe.text_encoder_2, "forward", "text_encoder_2")
        self._instrument(self.pipe.unet, "forward", "unet_step")
        self._instrument(self.pipe.vae, "decode", "vae_decode")

    def warm_up(self, prompt: str = "A warm cache", save: bool = True):
        """Run one untimed generation: CUDA context, cuDNN autotuning, allocator caches."""
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
