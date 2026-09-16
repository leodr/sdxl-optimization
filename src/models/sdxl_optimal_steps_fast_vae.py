from contextlib import contextmanager
from functools import wraps
import gc
import time

from diffusers import DiffusionPipeline
from PIL import Image
from pruna import SmashConfig, smash
import torch
from torch.profiler import record_function

from models.sdxl_base import Timings
from models.sdxl_base_vae_bf16 import cast_vae_decode
from models.sdxl_pruna import PRUNA_CACHE_DIR
from output import save_with_prompt


class SDXLOptimalStepsFastVAE:
    """SDXL base 1.0 in fp16, sampling with a fixed list of timesteps (e.g. from the OSS search), with pruna
    torch.compile on the U-Net, the VAE decoder in bfloat16 with torch.compile on vae.decode, no invisible
    watermark, and per-component timing.

    Identical to SDXLOptimalStepsNVFP4FastVAE without NVFP4: the U-Net stays fp16. The bf16 decoder is set up as in
    SDXLBaseVAEBF16; add_watermarker=False saves ~48 ms of CPU time per image. Without DeepCache, pruna's
    torch_compile compiles unet.forward as a whole; it does not compile vae.decode, so that is compiled here."""

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, timesteps: list[int], schedule_source: str | None = None,
                 compile_mode: str = "max-autotune-no-cudagraphs", vae_compile_mode: str = "max-autotune-no-cudagraphs",
                 device: str = "cuda"):
        """timesteps: descending integer timesteps, one per U-Net step, e.g. [981, 941, ..., 1].
        schedule_source: where they come from, for run records.
        compile_mode / vae_compile_mode: torch.compile modes for the U-Net (via pruna) and for vae.decode."""
        self.timesteps = list(timesteps)
        self.schedule_source = schedule_source
        self.smash_config = {"torch_compile": True, "torch_compile_mode": compile_mode}
        self.vae_compile_mode = vae_compile_mode
        self.device = device
        self.pipe = None
        # Per-label durations of the generation being timed; None while not timing.
        self._timings: dict[str, list[float]] | None = None
        # Known after prepare().
        self._resolved_smash_config: str | None = None

    def describe(self) -> dict:
        """What exactly this model is, for run records."""
        return {
            "class": type(self).__name__,
            "model_id": self.model_id,
            "dtype": "float16",
            "timesteps": self.timesteps,
            "schedule_source": self.schedule_source,
            "smash_config": self.smash_config,
            "smash_config_resolved": self._resolved_smash_config,
            "vae_decode_compile_mode": self.vae_compile_mode,
            "vae_decode_dtype": "bfloat16",
            "watermark": False,
        }

    def prepare(self):
        """Load the pipeline, compile the U-Net (pruna) and the bf16 vae.decode, then instrument its components."""
        pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
                                                 add_watermarker=False)
        pipe.to(self.device)
        pipe.vae.to(torch.bfloat16)

        config = SmashConfig(self.smash_config, device=self.device, cache_dir_prefix=PRUNA_CACHE_DIR)
        config.disable_saving()  # inference only; don't keep copies of the model around for saving
        self.pipe = smash(model=pipe, smash_config=config)
        self._resolved_smash_config = str(config)
        print(f"smashed with {config}")

        # With a bf16 VAE the pipeline no longer upcasts; the cast wrapper moves the fp16 latents to bf16 and the image
        # back to float32, and is compiled together with the decode.
        cast_vae_decode(self.pipe.vae, torch.bfloat16)
        self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode=self.vae_compile_mode)

        # Instrument after compiling, so the timing wrappers sit outside the compiled regions.

        self._instrument(self.pipe.text_encoder, "forward", "text_encoder")
        self._instrument(self.pipe.text_encoder_2, "forward", "text_encoder_2")
        self._instrument(self.pipe.unet, "forward", "unet_step")
        self._instrument(self.pipe.vae, "decode", "vae_decode")

    def warm_up(self, prompt: str = "A warm cache", save: bool = True):
        """Run one untimed generation: compilation, CUDA context, cuDNN autotuning, allocator caches."""
        image = self.pipe(prompt=prompt, timesteps=self.timesteps).images[0]
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
                image = self.pipe(prompt=prompt, timesteps=self.timesteps, generator=generator).images[0]
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
