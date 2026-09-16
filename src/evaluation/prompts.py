import json
import random

from huggingface_hub import hf_hub_download

DATASET = "playgroundai/MJHQ-30K"


def sample_prompts(n: int = 50, seed: int = 0) -> list[str]:
    """Sample n prompts from MJHQ-30K, spread evenly over its 10 categories, in shuffled order.

    Only the prompt metadata (meta_data.json, ~9 MB) is downloaded, not the images. The same seed gives
    the same prompts.
    """
    path = hf_hub_download(DATASET, "meta_data.json", repo_type="dataset")
    by_category: dict[str, list[str]] = {}
    for key, entry in sorted(json.load(open(path)).items()):
        by_category.setdefault(entry["category"], []).append(entry["prompt"])

    rng = random.Random(seed)
    categories = sorted(by_category)
    per_category, extra = divmod(n, len(categories))
    prompts = []
    for i, category in enumerate(categories):
        prompts += rng.sample(by_category[category], per_category + (1 if i < extra else 0))
    rng.shuffle(prompts)
    return prompts
