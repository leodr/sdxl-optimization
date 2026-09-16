"""Reference images: generate the base model's images once, then compare other models against them."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
import socket

from PIL import Image
import torch

from evaluation.compare import format_results, generate_all, score_pairs, summarize_metrics, summarize_timings
from models.sdxl_base import Timings
from output import make_grid

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
REFERENCES_DIR = PROJECT_DIR / "references"
RUNS_DIR = PROJECT_DIR / "runs"
GRID_SIZE = 16  # the first 16 prompts, as a 4x4 grid
GRID_JPEG_QUALITY = 90  # grid.jpg is the committed copy (~1 MB instead of ~6 MB); grid.png stays local


def save_grid(grid: Image.Image, directory: Path) -> Path:
    """Save grid.png (lossless, git-ignored) and grid.jpg (for the repository)."""
    grid.save(directory / "grid.png")
    grid.convert("RGB").save(directory / "grid.jpg", quality=GRID_JPEG_QUALITY)
    return directory / "grid.jpg"


def generate_reference(model, prompts: list[str], seed: int, name: str, overwrite: bool = False) -> Path:
    """Generate every prompt with the (unprepared) model and store the raw images in references/<name>/.

    manifest.json records the prompts, seed, timings and the host/GPU, so a later comparison uses the
    exact same settings. grid.png / grid.jpg show the first 16 images.
    """
    out_dir = REFERENCES_DIR / name
    if (out_dir / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already holds a reference; pass overwrite=True to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)

    images, timings = generate_all(model, prompts, seed, name)
    entries = []
    for i, (prompt, image, t) in enumerate(zip(prompts, images, timings)):
        filename = f"{i:03d}.png"
        image.save(out_dir / filename)
        entries.append({"file": filename, "prompt": prompt, "timings": asdict(t)})

    manifest = {
        "model": type(model).__name__,
        "description": model.describe(),
        "seed": seed,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "images": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_reference_grid(name)
    print(f"wrote {len(entries)} reference images to {out_dir}")
    return out_dir


def write_reference_grid(name: str) -> Path:
    """(Re)create references/<name>/grid.png and grid.jpg from the stored images."""
    ref_dir = REFERENCES_DIR / name
    manifest = json.loads((ref_dir / "manifest.json").read_text())
    entries = manifest["images"][:GRID_SIZE]
    total = summarize_timings([Timings(**e["timings"]) for e in manifest["images"]])["total_ms"]
    grid = make_grid([Image.open(ref_dir / e["file"]) for e in entries], [e["prompt"] for e in entries],
                     title=f"reference {name} ({manifest['model']}), {total:.0f} ms/image, seed {manifest['seed']}")
    return save_grid(grid, ref_dir)


def compare_to_reference(model, name: str, run_name: str) -> Path:
    """Generate the reference's prompts with the (unprepared) model, score it against references/<name>/,
    and write everything to a new folder runs/<timestamp>_<run_name>/:

    - run.json: model description, host/GPU, seed, metric and timing summaries, and per image the
      prompt, metrics and timings
    - results.txt: the printed report
    - grid.png / grid.jpg: the first 16 images as a 4x4 grid, captioned (compare with references/<name>/grid.jpg)
    - images/000.png ...: all generated images, without captions

    A is the reference, B the model. Returns the run folder.
    """
    ref_dir = REFERENCES_DIR / name
    manifest = json.loads((ref_dir / "manifest.json").read_text())
    prompts = [e["prompt"] for e in manifest["images"]]
    seed = manifest["seed"]

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_")
    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}"
    (run_dir / "images").mkdir(parents=True)
    print(f"run folder: {run_dir}")

    gpu = torch.cuda.get_device_name()
    warnings = []
    if gpu != manifest["gpu"]:
        warnings.append(f"reference was timed on {manifest['gpu']}, this is {gpu}; the speedup is not comparable")
        print(f"warning: {warnings[-1]}")

    images, timings = generate_all(model, prompts, seed, run_name)
    for i, image in enumerate(images):
        image.save(run_dir / "images" / f"{i:03d}.png")

    reference_images = [Image.open(ref_dir / e["file"]) for e in manifest["images"]]
    results = score_pairs(reference_images, images)
    timings_ref = summarize_timings([Timings(**e["timings"]) for e in manifest["images"]])
    timings_run = summarize_timings(timings)

    report = "\n".join([
        f"run {run_name}: {model.describe()}",
        f"A = reference {name} ({manifest['model']}), B = {run_name} ({type(model).__name__})",
        *[f"warning: {w}" for w in warnings],
        "",
        format_results(results, seed, timings_ref, timings_run),
    ])
    print("\n" + report)
    (run_dir / "results.txt").write_text(report + "\n")

    save_grid(make_grid(images[:GRID_SIZE], prompts[:GRID_SIZE],
                        title=f"{run_name} ({type(model).__name__}), {timings_run['total_ms']:.0f} ms/image "
                              f"({timings_ref['total_ms'] / timings_run['total_ms']:.2f}x), seed {seed}"), run_dir)

    record = {
        "run_name": run_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": model.describe(),
        "reference": name,
        "host": socket.gethostname(),
        "gpu": gpu,
        "seed": seed,
        "warnings": warnings,
        "metrics": summarize_metrics(results),
        "timings_mean_ms": timings_run,
        "reference_timings_mean_ms": timings_ref,
        "speedup": timings_ref["total_ms"] / timings_run["total_ms"],
        "images": [
            {
                "file": f"images/{i:03d}.png",
                "prompt": prompt,
                "metrics": {k: results[k][i] for k in results},
                "timings": asdict(t),
            }
            for i, (prompt, t) in enumerate(zip(prompts, timings))
        ],
    }
    # json.dumps writes inf as the non-standard Infinity; store identical pairs' PSNR as null instead.
    for entry in record["images"]:
        if entry["metrics"]["psnr"] == float("inf"):
            entry["metrics"]["psnr"] = None
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {run_dir}")
    return run_dir
