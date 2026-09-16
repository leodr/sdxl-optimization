import gc
import math
import statistics

from PIL import Image
import torch

from evaluation.metrics import PairMetrics
from models.sdxl_base import Timings


def generate_all(model, prompts: list[str], seed: int, name: str) -> tuple[list[Image.Image], list[Timings]]:
    """Prepare and warm up the model, generate every prompt with the same seed, then unload it.

    Images are returned, not saved; the caller decides where they go.
    """
    model.prepare()
    model.pipe.set_progress_bar_config(disable=True)
    model.warm_up(save=False)
    images, timings = [], []
    for i, prompt in enumerate(prompts):
        image, t = model.generate_timed(prompt, seed=seed, save=False)
        images.append(image)
        timings.append(t)
        print(f"[{name}] {i + 1}/{len(prompts)}  {t.total_ms:.0f} ms  {prompt[:60]!r}")
    model.unload()
    return images, timings


def score_pairs(images_a: list[Image.Image], images_b: list[Image.Image], device: str = "cuda") -> dict[str, list[float]]:
    """LPIPS, PSNR and DreamSim for each image pair. Loads the metric models and frees them afterwards."""
    metrics = PairMetrics(device)
    results = {"lpips": [], "psnr": [], "dreamsim": []}
    for a, b in zip(images_a, images_b, strict=True):
        for name, value in metrics(a, b).items():
            results[name].append(value)
    del metrics
    gc.collect()
    torch.cuda.empty_cache()
    return results


def summarize_metrics(results: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Mean, std, min and max per metric. PSNR excludes identical pairs (inf), which are counted separately."""
    summary = {}
    for name, values in results.items():
        finite = [v for v in values if math.isfinite(v)]
        if finite:
            summary[name] = {
                "mean": statistics.mean(finite),
                "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
                "min": min(finite),
                "max": max(finite),
            }
    summary["identical_pairs"] = sum(1 for v in results["psnr"] if not math.isfinite(v))
    return summary


def summarize_timings(timings: list[Timings]) -> dict[str, float]:
    """Mean per image of each timed component, in ms."""
    fields = ["text_encoder_ms", "text_encoder_2_ms", "unet_total_ms", "unet_step_median_ms", "vae_decode_ms",
              "other_ms", "total_ms"]
    return {f: statistics.mean(getattr(t, f) for t in timings) for f in fields}


def format_results(results: dict[str, list[float]], seed: int, timings_a: dict[str, float],
                   timings_b: dict[str, float]) -> str:
    """Metrics table and per-component timing comparison. timings_* come from summarize_timings."""
    summary = summarize_metrics(results)
    lines = [f"{len(results['psnr'])} image pairs, seed {seed}",
             f"{'metric':<28}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}"]
    for key, label in [("lpips", "lpips (lower = closer)"), ("dreamsim", "dreamsim (lower = closer)"),
                       ("psnr", "psnr dB (higher = closer)")]:
        if key in summary:
            s = summary[key]
            lines.append(f"{label:<28}{s['mean']:>10.4f}{s['std']:>10.4f}{s['min']:>10.4f}{s['max']:>10.4f}")
    lines.append(f"identical pairs (psnr = inf, excluded above): {summary['identical_pairs']}")

    lines += ["", f"{'timing, mean per image (ms)':<28}{'A':>10}{'B':>10}{'A/B':>10}"]
    for key, label in [("text_encoder_ms", "text_encoder"), ("text_encoder_2_ms", "text_encoder_2"),
                       ("unet_total_ms", "unet, all steps"), ("unet_step_median_ms", "unet step (median)"),
                       ("vae_decode_ms", "vae decode"), ("other_ms", "other"), ("total_ms", "total")]:
        a, b = timings_a[key], timings_b[key]
        lines.append(f"{label:<28}{a:>10.1f}{b:>10.1f}{a / b:>9.2f}x")
    return "\n".join(lines)


def compare_models(model_a, model_b, prompts: list[str], seed: int = 0) -> dict[str, list[float]]:
    """Generate all prompts with both models and print LPIPS, PSNR and DreamSim over the image pairs.

    The models are loaded one after the other, so pass them unprepared. Returns the per-pair metrics.
    """
    images_a, timings_a = generate_all(model_a, prompts, seed, "A")
    images_b, timings_b = generate_all(model_b, prompts, seed, "B")
    results = score_pairs(images_a, images_b)
    print(format_results(results, seed, summarize_timings(timings_a), summarize_timings(timings_b)))
    return results
