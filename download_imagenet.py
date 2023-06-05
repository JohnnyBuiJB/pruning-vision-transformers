import torch
from datasets import load_dataset, list_datasets
import os

DATA_PATH = '.cache'

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

dset = load_dataset(path='imagenet-1k', split='train', use_auth_token=True,
                    cache_dir=DATA_PATH, num_proc=4)

print(len(dset))