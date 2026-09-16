"""Write run_comparison.csv: the metrics and timings from each selected run's results, one row per run.

uv run python src/summarize_runs.py
"""

import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS = [
    "20260915_142608_base",
    "20260916_124552_base_vae_bf16",
    "20260916_143415_torch_compile_vae_bf16",
    "20260916_180719_nvfp4_torch_compile_vae_bf16",
    "20260915_144030_deepcache_torch_compile",
    "20260915_150024_deepcache_interval3_torch_compile",
    "20260915_173434_deepcache_interval3_torch_compile_nvfp4",
    "20260915_170608_optimal_steps_10",
    "20260915_181729_optimal_steps_20",
    "20260916_111502_optimal_steps_10_nvfp4_torch_compile_vae",
    "20260916_113946_optimal_steps_15_nvfp4_torch_compile_vae",
    "20260916_114302_optimal_steps_20_nvfp4_torch_compile_vae",
    "20260916_130107_optimal_steps_20_nvfp4_torch_compile_vae_bf16_no_watermark",
]
# Mean per image, in ms, as in results.txt. The text encoders are left out: they take ~7 ms in every run.
TIMING_FIELDS = ["unet_total_ms", "unet_step_median_ms", "vae_decode_ms", "other_ms", "total_ms"]


def main():
    rows = []
    for run in RUNS:
        r = json.loads((PROJECT_DIR / "runs" / run / "run.json").read_text())
        row = {"run_name": r["run_name"]}
        for metric in ("lpips", "dreamsim", "psnr"):
            for stat in ("mean", "std", "min", "max"):
                # `or 0.0` turns -0.0 from tiny negative distances into 0.0. PSNR is empty when every pair is identical.
                row[f"{metric}_{stat}"] = (round(r["metrics"][metric][stat], 4) or 0.0) if metric in r["metrics"] else ""
        row["identical_pairs"] = r["metrics"]["identical_pairs"]
        row.update({field: round(r["timings_mean_ms"][field], 1) for field in TIMING_FIELDS})
        row["speedup"] = round(r["speedup"], 2)  # total time of the reference / total time of this run
        rows.append(row)

    out = PROJECT_DIR / "run_comparison.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs to {out}")


if __name__ == "__main__":
    main()
