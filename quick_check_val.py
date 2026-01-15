import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import load_dataset

hf_val = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    streaming=False,
    cache_dir="/mnt/imagenet/hf_cache"
)

print("Loaded:", len(hf_val), "samples")
