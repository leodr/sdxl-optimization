"""Side-by-side speed video: the optimized config (left) against the base model (right).

Each side shows its images in real time: an image appears at the moment it finished generating in the timed run,
so every image stays on screen for as long as the next one took to generate.

    uv run python src/make_speed_video.py generate base 5        # on a GPU host, one config per process
    uv run python src/make_speed_video.py generate optimized 16
    uv run python src/make_speed_video.py render                 # CPU only, writes speed_comparison.mp4
"""

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_DIR / "video_frames"
VIDEO_PATH = PROJECT_DIR / "speed_comparison.mp4"
PROMPT = 'A post-it note with "{0}" written out on it. "{0}"'
SEED = 0

# Left to right: the optimized config (OSS 20 steps, NVFP4, torch.compile, bf16 VAE, no watermark), then base SDXL.
CONFIGS = ["optimized", "base"]

# Video layout.
W, H, FPS, DURATION_S = 1920, 1080, 30, 30.0
IMAGE_SIZE, GAP = 880, 80
PANEL_X = {"optimized": (W - 2 * IMAGE_SIZE - GAP) // 2, "base": (W + GAP) // 2}
BAR_GAP, BAR_H, CAPTION_GAP, CAPTION_SIZE = 14, 8, 22, 26
IMAGE_Y = (H - (IMAGE_SIZE + BAR_GAP + BAR_H + CAPTION_GAP + CAPTION_SIZE)) // 2
BAR_Y = IMAGE_Y + IMAGE_SIZE + BAR_GAP
CAPTION_Y = BAR_Y + BAR_H + CAPTION_GAP
BG, PANEL, TEXT2, BAR = (17, 17, 17), (38, 38, 38), (160, 160, 160), (225, 225, 225)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def generate(config: str, count: int):
    """Generate `count` images with one config (after its usual warm-up) and record each image's total time."""
    sys.path.insert(0, str(PROJECT_DIR / "src"))
    from models import SDXLBase, SDXLOptimalStepsNVFP4FastVAE
    from search_optimal_steps import schedule_path

    if config == "base":
        model = SDXLBase()
    else:
        timesteps = json.loads(schedule_path(50, 20).read_text())["timesteps"]
        model = SDXLOptimalStepsNVFP4FastVAE(timesteps, schedule_source=schedule_path(50, 20).name)
    model.prepare()
    model.pipe.set_progress_bar_config(disable=True)
    model.warm_up(save=False)

    out_dir = FRAMES_DIR / config
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(1, count + 1):
        prompt = PROMPT.format(i)
        image, timings = model.generate_timed(prompt, seed=SEED, save=False)
        image.save(out_dir / f"{i:02d}.png")
        records.append({"file": f"{i:02d}.png", "prompt": prompt, "total_ms": timings.total_ms})
        print(f"[{config}] {i}/{count} {timings.total_ms:.0f} ms", flush=True)
    (out_dir / "timings.json").write_text(json.dumps({"config": model.describe(), "images": records}, indent=2))


def render():
    font = ImageFont.truetype(FONT, CAPTION_SIZE)
    sides = {}
    for config in CONFIGS:
        records = json.loads((FRAMES_DIR / config / "timings.json").read_text())["images"]
        done_at, t = [], 0.0
        for r in records:
            t += r["total_ms"] / 1000
            done_at.append(t)
        if done_at[-1] < DURATION_S:
            raise SystemExit(f"{config}: images only cover {done_at[-1]:.1f} s of the {DURATION_S:.0f} s video; generate more")
        mean_s = sum(r["total_ms"] for r in records) / len(records) / 1000
        images = [Image.open(FRAMES_DIR / config / r["file"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                  for r in records]
        sides[config] = {"records": records, "done_at": done_at, "mean_s": mean_s, "images": images}

    def static_panel(canvas: Image.Image, config: str, shown: int):
        """The image (or an empty panel before the first one) and its prompt."""
        side, draw, x = sides[config], ImageDraw.Draw(canvas), PANEL_X[config]
        if shown == 0:
            draw.rectangle((x, IMAGE_Y, x + IMAGE_SIZE, IMAGE_Y + IMAGE_SIZE), fill=PANEL)
        else:
            canvas.paste(side["images"][shown - 1], (x, IMAGE_Y))
            draw.text((x, CAPTION_Y), side["records"][shown - 1]["prompt"], font=font, fill=TEXT2)

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(VIDEO_PATH)],
        stdin=subprocess.PIPE)
    cache = {}
    for frame in range(int(DURATION_S * FPS)):
        t = frame / FPS
        shown = {c: sum(1 for d in sides[c]["done_at"] if d <= t) for c in CONFIGS}
        key = (shown["optimized"], shown["base"])
        if key not in cache:
            canvas = Image.new("RGB", (W, H), BG)
            for c in CONFIGS:
                static_panel(canvas, c, shown[c])
            cache[key] = canvas
        canvas = cache[key].copy()
        draw = ImageDraw.Draw(canvas)
        for c in CONFIGS:
            # Progress of the image currently being generated.
            done_at, start = sides[c]["done_at"], ([0.0] + sides[c]["done_at"])[shown[c]]
            fraction = (t - start) / (done_at[shown[c]] - start) if shown[c] < len(done_at) else 1.0
            x = PANEL_X[c]
            draw.rectangle((x, BAR_Y, x + IMAGE_SIZE, BAR_Y + BAR_H), fill=PANEL)
            draw.rectangle((x, BAR_Y, x + int(IMAGE_SIZE * min(fraction, 1.0)), BAR_Y + BAR_H), fill=BAR)
        ffmpeg.stdin.write(canvas.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise SystemExit("ffmpeg failed")
    for c in CONFIGS:
        print(f"{c}: {sum(1 for d in sides[c]['done_at'] if d <= DURATION_S)} images in {DURATION_S:.0f} s "
              f"({sides[c]['mean_s']:.2f} s per image)")
    print(f"wrote {VIDEO_PATH}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "generate" and sys.argv[2] in CONFIGS:
        generate(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1:] == ["render"]:
        render()
    else:
        sys.exit(__doc__)
