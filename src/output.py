from datetime import datetime
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
# Pillow's bundled default font lacks accented glyphs (é renders as a box); DejaVu Sans covers them.
DEJAVU_SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(DEJAVU_SANS, size) if DEJAVU_SANS.exists() else ImageFont.load_default(size=size)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy word wrap to max_width pixels."""
    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and font.getlength(candidate) > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def render_with_prompt(image: Image.Image, prompt: str, font_size: int = 32, padding: int = 24,
                       max_lines: int | None = None, min_text_lines: int = 0) -> Image.Image:
    """The image with the prompt rendered below it, black on white.

    max_lines truncates long prompts with an ellipsis; min_text_lines reserves space for that many lines,
    so captions of different lengths give equally tall results.
    """
    font = _font(font_size)
    max_width = image.width - 2 * padding
    lines = _wrap(prompt, font, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and font.getlength(last + " …") > max_width:
            last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
        lines[-1] = last + " …"

    ascent, descent = font.getmetrics()
    line_height = ascent + descent + font_size // 4
    text_height = max(len(lines), min_text_lines) * line_height + 2 * padding

    canvas = Image.new("RGB", (image.width, image.height + text_height), "white")
    canvas.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((padding, image.height + padding + i * line_height), line, fill="black", font=font)
    return canvas


def save_with_prompt(image: Image.Image, prompt: str, font_size: int = 32, padding: int = 24) -> Path:
    """Save the image with the prompt rendered below it, black on white, into outputs/."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:50]
    path = OUTPUTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.png"
    render_with_prompt(image, prompt, font_size, padding).save(path)
    return path


def make_grid(images: list[Image.Image], prompts: list[str], title: str, cols: int = 4, cell_size: int = 512,
              gap: int = 8, caption_lines: int = 3) -> Image.Image:
    """A grid of images with truncated prompt captions and a title bar, for side-by-side comparison."""
    cells = [
        render_with_prompt(image.resize((cell_size, cell_size), Image.Resampling.LANCZOS), prompt,
                           font_size=18, padding=8, max_lines=caption_lines, min_text_lines=caption_lines)
        for image, prompt in zip(images, prompts, strict=True)
    ]
    rows = -(-len(cells) // cols)
    cell_w, cell_h = cells[0].size

    title_font = _font(28)
    title_lines = _wrap(title, title_font, cols * cell_w)
    ascent, descent = title_font.getmetrics()
    title_h = len(title_lines) * (ascent + descent + 6) + 2 * gap

    grid = Image.new("RGB", (cols * cell_w + (cols + 1) * gap, title_h + rows * cell_h + rows * gap), "white")
    draw = ImageDraw.Draw(grid)
    for i, line in enumerate(title_lines):
        draw.text((gap, gap + i * (ascent + descent + 6)), line, fill="black", font=title_font)
    for i, cell in enumerate(cells):
        row, col = divmod(i, cols)
        grid.paste(cell, (gap + col * (cell_w + gap), title_h + row * (cell_h + gap)))
    return grid
