"""Pareto chart of speedup vs mean LPIPS over all runs: pareto_speed_lpips.png / .svg in the project root.

uv run --with matplotlib python src/plot_pareto.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
# Nimbus Sans is URW's metric-compatible Helvetica; fall back to DejaVu Sans if it's not installed.
NIMBUS_SANS = Path("/usr/share/fonts/opentype/urw-base35")

SURFACE, TEXT, TEXT2, MUTED, GRID, ACCENT = "#ffffff", "#0b0b0b", "#52514e", "#a3a29d", "#e4e3df", "#2a78d6"

# The recommended configurations: ringed markers and bold labels.
RECOMMENDED = {
    "20260916_143415_torch_compile_vae_bf16",
    "20260916_234736_optimal_steps_20_torch_compile_vae_bf16_no_watermark",
    "20260916_130107_optimal_steps_20_nvfp4_torch_compile_vae_bf16_no_watermark",
}

# Short labels for the runs worth naming; runs not listed are plotted without a label.
# Values: (label, x offset in points, y offset in points, horizontal alignment).
LABELS = {
    "20260915_142608_base": ("Base (50 steps)", 8, -12, "left"),
    "20260916_143415_torch_compile_vae_bf16": ("compile + bf16 VAE", -8, 34, "center"),
    "20260916_180719_nvfp4_torch_compile_vae_bf16": ("NVFP4 + compile + bf16 VAE", 8, -12, "left"),
    "20260915_144030_deepcache_torch_compile": ("DeepCache i2 + compile", -8, 8, "right"),
    "20260915_150024_deepcache_interval3_torch_compile": ("DeepCache i3 + compile", -8, 12, "right"),
    "20260915_181729_optimal_steps_20": ("OSS 20", 6, 14, "left"),
    "20260916_234736_optimal_steps_20_torch_compile_vae_bf16_no_watermark": ("OSS 20 + compile + bf16 VAE", 12, -18, "left"),
    "20260915_175459_optimal_steps_15": ("OSS 15", 8, -6, "left"),
    "20260915_170608_optimal_steps_10": ("OSS 10", -8, 6, "right"),
    "20260915_170813_euler_10": ("Euler 10 steps (default schedule)", -8, 0, "right"),
    "20260916_130107_optimal_steps_20_nvfp4_torch_compile_vae_bf16_no_watermark": ("OSS 20 + NVFP4 + compile + bf16 VAE", 12, -22, "left"),
    "20260916_125805_optimal_steps_15_nvfp4_torch_compile_vae_bf16_no_watermark": ("OSS 15 + NVFP4 + compile + bf16 VAE", 0, 22, "center"),
    "20260916_152932_optimal_steps_20_deepcache_nvfp4_torch_compile_vae_bf16_no_watermark": ("OSS 20 + DeepCache\n+ NVFP4 + …", 8, -30, "center"),
    "20260916_125406_optimal_steps_10_nvfp4_torch_compile_vae_bf16_no_watermark": ("OSS 10 + NVFP4 + compile\n+ bf16 VAE", -4, 10, "right"),
    "20260915_164230_hyper": ("Hyper-SD (8 steps)", -8, 0, "right"),
}


def load_runs() -> list[dict]:
    runs = []
    for path in sorted((PROJECT_DIR / "runs").glob("*/run.json")):
        record = json.loads(path.read_text())
        runs.append({"run": path.parent.name, "speedup": record["speedup"], "lpips": record["metrics"]["lpips"]["mean"]})
    return runs


def pareto_front(runs: list[dict]) -> list[dict]:
    """Runs no other run beats on both axes (at least as fast and at least as close, strictly better on one)."""
    def dominated(r):
        return any(o["speedup"] >= r["speedup"] and o["lpips"] <= r["lpips"]
                   and (o["speedup"] > r["speedup"] or o["lpips"] < r["lpips"]) for o in runs)
    return sorted((r for r in runs if not dominated(r)), key=lambda r: r["speedup"])


def main():
    if NIMBUS_SANS.exists():
        for font in NIMBUS_SANS.glob("NimbusSans-*.otf"):
            font_manager.fontManager.addfont(str(font))
        family = "Nimbus Sans"
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 11, "svg.fonttype": "path"})

    runs = load_runs()
    front = pareto_front(runs)
    front_names = {r["run"] for r in front}

    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT2, length=0)

    ax.plot([r["speedup"] for r in front], [r["lpips"] for r in front], color=ACCENT, lw=2, zorder=2,
            label="Pareto frontier")
    others = [r for r in runs if r["run"] not in front_names]
    ax.scatter([r["speedup"] for r in others], [r["lpips"] for r in others], s=64, color=MUTED,
               edgecolor=SURFACE, linewidth=2, zorder=3, label="other runs")
    ax.scatter([r["speedup"] for r in front], [r["lpips"] for r in front], s=72, color=ACCENT,
               edgecolor=SURFACE, linewidth=2, zorder=4, label="on the frontier")
    recommended = [r for r in runs if r["run"] in RECOMMENDED]
    ax.scatter([r["speedup"] for r in recommended], [r["lpips"] for r in recommended], s=300, facecolor="none",
               edgecolor=TEXT, linewidth=1.8, zorder=5, label="recommended")

    for r in runs:
        if r["run"] in LABELS:
            text, dx, dy, ha = LABELS[r["run"]]
            ax.annotate(text, (r["speedup"], r["lpips"]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        va="center", fontsize=10, color=TEXT if r["run"] in front_names else TEXT2, zorder=6,
                        fontweight="bold" if r["run"] in RECOMMENDED else "normal")

    ax.set_xlim(0.6, 7.6)
    ax.set_ylim(-0.03, 0.66)
    ax.set_xlabel("Speedup over base SDXL (×, higher is faster)", color=TEXT2, fontsize=11, labelpad=10)
    ax.set_ylabel("Mean LPIPS vs base images (lower is closer)", color=TEXT2, fontsize=11, labelpad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=TEXT2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.1)

    for suffix in ("png", "svg"):
        fig.savefig(PROJECT_DIR / f"pareto_speed_lpips.{suffix}", facecolor=SURFACE)
    print("frontier:", ", ".join(r["run"].split("_", 2)[2] for r in front))


if __name__ == "__main__":
    main()
