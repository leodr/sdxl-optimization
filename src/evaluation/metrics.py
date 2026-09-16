import math

from dreamsim import dreamsim
import lpips
import numpy as np
from PIL import Image
import torch

# dreamsim defaults to ./models; keep its weights with the other torch caches on local disk. Resolved
# at import because dreamsim calls torch.hub.set_dir(cache_dir), which would nest a later lookup.
DREAMSIM_CACHE = f"{torch.hub.get_dir()}/dreamsim"


def psnr(a: Image.Image, b: Image.Image) -> float:
    """PSNR in dB over RGB pixels; inf for identical images."""
    x = np.asarray(a.convert("RGB"), dtype=np.float64) / 255
    y = np.asarray(b.convert("RGB"), dtype=np.float64) / 255
    mse = np.mean((x - y) ** 2)
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)


def _to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    """PIL image to a (1, 3, H, W) float tensor in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


class PairMetrics:
    """LPIPS (AlexNet), PSNR and DreamSim (ensemble) for image pairs. Lower LPIPS/DreamSim is more similar."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.dreamsim, self.dreamsim_preprocess = dreamsim(pretrained=True, device=device, cache_dir=DREAMSIM_CACHE)

    @torch.no_grad()
    def __call__(self, a: Image.Image, b: Image.Image) -> dict[str, float]:
        return {
            "lpips": self.lpips(_to_tensor(a, self.device), _to_tensor(b, self.device), normalize=True).item(),
            "psnr": psnr(a, b),
            "dreamsim": self.dreamsim(
                self.dreamsim_preprocess(a).to(self.device), self.dreamsim_preprocess(b).to(self.device)
            ).item(),
        }
