"""Compare one model variant against the base reference: uv run python src/evaluate.py <variant>

Run each variant in its own process, so torch.compile caches from one variant can't affect the next.
"""

import json
import sys

from evaluation import compare_to_reference
from models import SDXLBase, SDXLBaseVAEBF16, SDXLCompiledVAEBF16, SDXLNVFP4CompiledVAEBF16, SDXLOptimalSteps, SDXLOptimalStepsDeepCacheNVFP4FastVAE, SDXLOptimalStepsFastVAE, SDXLOptimalStepsNVFP4, SDXLOptimalStepsNVFP4FastVAE, SDXLPruna, SDXLPrunaCompiledVAE, SDXLPrunaNVFP4
from search_optimal_steps import schedule_path


def optimal_steps(teacher_steps: int, student_steps: int, model_class=SDXLOptimalSteps):
    path = schedule_path(teacher_steps, student_steps)
    if not path.exists():
        sys.exit(f"{path} is missing; run src/search_optimal_steps.py --teacher-steps {teacher_steps} "
                 f"--student-steps {student_steps} first")
    return model_class(json.loads(path.read_text())["timesteps"], schedule_source=str(path.name))


VARIANTS = {
    "base": lambda: SDXLBase(),
    "torch_compile": lambda: SDXLPruna(["torch_compile"]),
    "deepcache_torch_compile": lambda: SDXLPruna(["deepcache", "torch_compile"]),
    # max-autotune (with CUDA graphs) fails with DeepCache: the cached features are CUDA graph outputs that
    # the next graph run overwrites ("accessing tensor output of CUDAGraphs that has been overwritten").
    "deepcache_torch_compile_max_autotune_no_cudagraphs": lambda: SDXLPruna(
        {"deepcache": True, "torch_compile": True, "torch_compile_mode": "max-autotune-no-cudagraphs"}),
    "deepcache_interval3_torch_compile": lambda: SDXLPruna(
        {"deepcache": True, "deepcache_interval": 3, "torch_compile": True}),
    "deepcache_torch_compile_max_autotune_no_cudagraphs_vae": lambda: SDXLPrunaCompiledVAE(
        {"deepcache": True, "torch_compile": True, "torch_compile_mode": "max-autotune-no-cudagraphs"},
        vae_compile_mode="max-autotune-no-cudagraphs"),
    # padding_pruning is not applicable: pruna requires a max_sequence_length pipeline argument (T5-based
    # pipelines like Flux/SD3); SDXL's CLIP encoders use a fixed 77 tokens.
    "deepcache_qkv_torch_compile_max_autotune_no_cudagraphs_vae": lambda: SDXLPrunaCompiledVAE(
        {"deepcache": True, "qkv_diffusers": True, "torch_compile": True,
         "torch_compile_mode": "max-autotune-no-cudagraphs"},
        vae_compile_mode="max-autotune-no-cudagraphs"),
    # Hyper-SDXL 8-step LoRA with the TCD scheduler; pruna sets num_inference_steps=8 and guidance_scale=0.
    "hyper": lambda: SDXLPruna(["hyper"]),
    # OSS (vendored OptimalSteps): 10 of the reference's 50 Euler timesteps, chosen to follow its trajectory.
    "optimal_steps_10": lambda: optimal_steps(50, 10),
    "optimal_steps_15": lambda: optimal_steps(50, 15),
    "optimal_steps_20": lambda: optimal_steps(50, 20),
    # Baseline for optimal_steps_10: the scheduler's default 10-step schedule, same as num_inference_steps=10.
    "euler_10": lambda: SDXLOptimalSteps(list(range(901, 0, -100)), schedule_source="default 10-step Euler"),
    # Same pruna config as deepcache_interval3_torch_compile, plus NVFP4 on selected U-Net linears.
    "deepcache_interval3_torch_compile_nvfp4": lambda: SDXLPrunaNVFP4(
        {"deepcache": True, "deepcache_interval": 3, "torch_compile": True}),
    # OSS schedule + NVFP4 + torch.compile (max-autotune-no-cudagraphs) on the U-Net and on vae.decode, no DeepCache.
    "optimal_steps_10_nvfp4_torch_compile_vae": lambda: optimal_steps(50, 10, SDXLOptimalStepsNVFP4),
    "optimal_steps_15_nvfp4_torch_compile_vae": lambda: optimal_steps(50, 15, SDXLOptimalStepsNVFP4),
    "optimal_steps_20_nvfp4_torch_compile_vae": lambda: optimal_steps(50, 20, SDXLOptimalStepsNVFP4),
    # The base model with only the VAE decoder moved from float32 to bf16.
    "base_vae_bf16": lambda: SDXLBaseVAEBF16(),
    # Base model (50 steps, watermark on) with torch.compile on the U-Net and on the bf16 VAE decode.
    "torch_compile_vae_bf16": lambda: SDXLCompiledVAEBF16(),
    # torch_compile_vae_bf16 plus NVFP4 on the U-Net: isolates NVFP4 at 50 steps.
    "nvfp4_torch_compile_vae_bf16": lambda: SDXLNVFP4CompiledVAEBF16(),
    # As optimal_steps_*_nvfp4_torch_compile_vae, with the compiled VAE decoder in bf16 and no invisible watermark.
    "optimal_steps_10_nvfp4_torch_compile_vae_bf16_no_watermark": lambda: optimal_steps(50, 10, SDXLOptimalStepsNVFP4FastVAE),
    "optimal_steps_15_nvfp4_torch_compile_vae_bf16_no_watermark": lambda: optimal_steps(50, 15, SDXLOptimalStepsNVFP4FastVAE),
    "optimal_steps_20_nvfp4_torch_compile_vae_bf16_no_watermark": lambda: optimal_steps(50, 20, SDXLOptimalStepsNVFP4FastVAE),
    # OSS 20 steps + torch.compile + compiled bf16 VAE decode, no watermark, no NVFP4.
    "optimal_steps_20_torch_compile_vae_bf16_no_watermark": lambda: optimal_steps(50, 20, SDXLOptimalStepsFastVAE),
    # The 20-step fast-VAE configuration plus DeepCache (interval 2) via pruna.
    "optimal_steps_20_deepcache_nvfp4_torch_compile_vae_bf16_no_watermark":
        lambda: optimal_steps(50, 20, SDXLOptimalStepsDeepCacheNVFP4FastVAE),
}

if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
    sys.exit(f"usage: evaluate.py <variant>, one of: {', '.join(VARIANTS)}")

variant = sys.argv[1]
compare_to_reference(VARIANTS[variant](), name="sdxl_base", run_name=variant)
