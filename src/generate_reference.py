"""One-off: generate the base model's reference images for the evaluation prompts."""

from evaluation import generate_reference, sample_prompts
from models import SDXLBase

prompts = sample_prompts(n=50, seed=0)
generate_reference(SDXLBase(), prompts, seed=0, name="sdxl_base")
