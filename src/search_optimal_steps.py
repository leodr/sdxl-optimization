"""Search an OSS timestep schedule for SDXL base: uv run python src/search_optimal_steps.py [--student-steps 10]

For each calibration prompt, runs the teacher (the reference's 50-step Euler schedule, guidance 5.0) and the
OSS search (see oss_sdxl.py), then takes the per-position median over prompts, as the vendored DiT example
does. The calibration prompts are MJHQ-30K prompts outside the evaluation set, with a different noise seed,
so the schedule is not fitted to the images it is evaluated on.

Writes schedules/optimal_steps_sdxl_base_<teacher>to<student>.json, which evaluate.py reads.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import time

from diffusers import DiffusionPipeline
import torch

from evaluation import sample_prompts
from oss_sdxl import check_adaptation, search_prompt
from vendor.optimal_steps.OSS.OSS import cal_medium

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCHEDULES_DIR = PROJECT_DIR / "schedules"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def schedule_path(teacher_steps: int, student_steps: int) -> Path:
    return SCHEDULES_DIR / f"optimal_steps_sdxl_base_{teacher_steps}to{student_steps}.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-steps", type=int, default=10)
    parser.add_argument("--teacher-steps", type=int, default=50)
    parser.add_argument("--prompts", type=int, default=10, help="number of calibration prompts")
    parser.add_argument("--prompt-seed", type=int, default=1, help="seed for sampling calibration prompts")
    parser.add_argument("--noise-seed", type=int, default=1, help="generation seed for the calibration images")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--check", action="store_true",
                        help="first verify on one prompt that the search's steps match the pipeline's")
    args = parser.parse_args()

    evaluation_prompts = {e["prompt"] for e in
                          json.loads((PROJECT_DIR / "references/sdxl_base/manifest.json").read_text())["images"]}
    prompts = [p for p in sample_prompts(n=args.prompts, seed=args.prompt_seed) if p not in evaluation_prompts]
    if len(prompts) < args.prompts:
        print(f"warning: dropped {args.prompts - len(prompts)} calibration prompts that are in the evaluation set")

    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    common = dict(student_steps=args.student_steps, teacher_steps=args.teacher_steps, guidance_scale=args.guidance_scale)
    if args.check:
        diffs = check_adaptation(pipe, prompts[0], args.noise_seed, **common)
        print(f"check (max abs latent difference, 0 = identical): {diffs}")

    start = time.perf_counter()
    results = []
    for i, prompt in enumerate(prompts):
        r = search_prompt(pipe, prompt, args.noise_seed, **common)
        results.append(r)
        print(f"[{i + 1}/{len(prompts)}] {r['seconds']:.0f} s  mse {r['final_latent_mse']:.5f}  "
              f"timesteps {r['timesteps']}  {prompt[:50]!r}")

    oss_steps = cal_medium([r["oss_steps"] for r in results])
    pipe.scheduler.set_timesteps(args.teacher_steps)
    teacher_timesteps = [int(t) for t in pipe.scheduler.timesteps]
    timesteps = [teacher_timesteps[args.teacher_steps - i] for i in reversed(oss_steps)]
    assert timesteps == sorted(set(timesteps), reverse=True), timesteps
    print(f"\nmedian schedule: indices {oss_steps}, timesteps {timesteps}")

    record = {
        "timesteps": timesteps,
        "oss_steps": oss_steps,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "scheduler": type(pipe.scheduler).__name__,
        "teacher_steps": args.teacher_steps,
        "teacher_timesteps": teacher_timesteps,
        "student_steps": args.student_steps,
        "guidance_scale": args.guidance_scale,
        "prompt_seed": args.prompt_seed,
        "noise_seed": args.noise_seed,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "search_seconds": time.perf_counter() - start,
        "per_prompt": results,
    }
    path = schedule_path(args.teacher_steps, args.student_steps)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
