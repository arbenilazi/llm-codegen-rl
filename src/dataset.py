import os
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Type

import pandas as pd
import datasets as hf_datasets
from datasets import DatasetDict, load_from_disk
from pathlib import Path

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# Dataset registry.
#
# To add a new dataset:
# 1) create an adapter implementing the unified interface
# 2) register it with register_dataset(...)
# 3) import the adapter module so registration executes

DATASET_REGISTRY: Dict[str, Type["Dataset"]] = {}


def register_dataset(
    name: str,
    adapter_cls: Type["Dataset"],
    aliases: Optional[Iterable[str]] = None,
) -> None:
    """
    Register a dataset adapter under one or more canonical names.
    Names are matched case-insensitively.
    """
    keys = {name.lower()}
    if aliases:
        keys.update(alias.lower() for alias in aliases)
    for key in keys:
        DATASET_REGISTRY[key] = adapter_cls


class Dataset:
    """
    Abstract base class for a code-evaluation dataset.
    Implementations must produce a unified schema:
      id, problem_description, input, solution, entry_point, test
    """

    def __init__(self) -> None:
        pass

    def check(self, solutions: List[List[str]], split: str, ks: List[int]) -> pd.DataFrame:
        raise NotImplementedError

    def __iter__(self) -> Iterator:
        raise NotImplementedError

    def evaluate(self, predictions: pd.DataFrame, ks: List[int], split: str) -> pd.DataFrame:
        raise NotImplementedError


def prepare_dataset(
    dataset_name: str,
    split_ratio: float,
    output_dir: str,
    seed: int = 42,
):
    """
    Download an HF dataset by hub id, split into train/test, and save with save_to_disk.
    Used when --dataset_path doesn't exist yet.
    """
    ds = hf_datasets.load_dataset(dataset_name)["train"]
    print(f"Splitting dataset '{dataset_name}' into train/test using ratio {split_ratio}...")
    ds = ds.train_test_split(test_size=1 - split_ratio, seed=seed)

    print(f"Saving dataset to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    ds.save_to_disk(output_dir)
    print(f"Saved train and test splits to {output_dir}.")
    return ds


# Import adapters so they register themselves with the registry.
from adapters.mbpp_adapter import MBPPDataset  # noqa: F401,E402
from adapters.taco_adapter import TacoVerifiedDataset  # noqa: F401,E402


def get_dataset(dataset_name: str, dataset_path: str) -> Optional[Dataset]:
    key = (dataset_name or "").lower()
    adapter_cls = DATASET_REGISTRY.get(key)
    if adapter_cls is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    path = dataset_path
    if not path:
        default_hf_id = getattr(adapter_cls, "DEFAULT_HF_ID", None)
        if default_hf_id:
            path = default_hf_id
    if not path:
        raise ValueError(f"Dataset path is required for dataset: {dataset_name}")

    dataset_dir = Path(path).resolve()
    path_to_use = str(dataset_dir) if dataset_dir.exists() else path

    stitched_base: Optional[DatasetDict] = None
    stitched = False
    if dataset_dir.exists():
        try:
            base = load_from_disk(str(dataset_dir))
        except Exception:
            base = None
        else:
            if isinstance(base, DatasetDict):
                present = set(base.keys())
            else:
                base = DatasetDict({"train": base}) if base is not None else None
                present = set(base.keys()) if base is not None else set()

            if base is not None:
                for split_name in ("train", "test"):
                    view_dir = dataset_dir / split_name
                    if split_name not in present and view_dir.is_dir():
                        try:
                            view_ds = load_from_disk(str(view_dir))
                            if isinstance(view_ds, DatasetDict):
                                if split_name in view_ds:
                                    base[split_name] = view_ds[split_name]
                                    stitched = True
                                    print(f"[i] stitched missing split '{split_name}' from on-disk view")
                                else:
                                    # fall back to single dataset stored under default key
                                    inner_keys = list(view_ds.keys())
                                    if inner_keys:
                                        base[split_name] = view_ds[inner_keys[0]]
                                        stitched = True
                                        print(f"[i] stitched missing split '{split_name}' from on-disk view")
                        except Exception as exc:
                            print(f"[warn] failed to stitch split '{split_name}' from {view_dir}: {exc}")
                stitched_base = base if stitched else None

    if not stitched or stitched_base is None:
        return adapter_cls(datapath=path_to_use)

    dataset_dir_str = str(dataset_dir)
    original_loaders = []

    def _patched_loader(path_arg, *args, **kwargs):
        try:
            resolved = str(Path(path_arg).resolve())
        except Exception:
            resolved = str(path_arg)
        if resolved == dataset_dir_str:
            return stitched_base
        return load_from_disk(path_arg, *args, **kwargs)

    modules_to_patch = {adapter_cls.__module__, "datasets"}
    for mod_name in modules_to_patch:
        module = sys.modules.get(mod_name)
        if module and hasattr(module, "load_from_disk"):
            original_loaders.append((module, module.load_from_disk))
            module.load_from_disk = _patched_loader

    try:
        return adapter_cls(datapath=path_to_use)
    finally:
        for module, original in original_loaders:
            module.load_from_disk = original


__all__ = ["Dataset", "prepare_dataset", "register_dataset", "get_dataset"]
