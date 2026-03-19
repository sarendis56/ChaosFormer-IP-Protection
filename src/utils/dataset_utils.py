import torch
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, cast
from torchvision import transforms, datasets

def _make_transforms(processor):
    """Standard transforms for vision classification."""
    # Use processor's normalization if available, else standard ImageNet stats
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, test_transform

def _wrap_hf_dataset(hf_split: Any, transform, label_key: str = "label", image_key: str = "image", label_offset: int = 0):
    class HFDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, split: Any):
            self.split = split
        def __len__(self) -> int:
            return int(len(self.split))
        def __getitem__(self, idx: int):
            ex = self.split[idx]
            image = ex[image_key].convert("RGB")
            label = int(ex[label_key]) + int(label_offset)
            if transform:
                image = transform(image)
            return image, label
    return HFDatasetWrapper(hf_split)

def _infer_label_offset(hf_split: Any, label_key: str = "label", sample_n: int = 100) -> int:
    """Infer whether labels are 1-indexed and should be shifted by -1."""
    n = int(len(hf_split))
    if n <= 0:
        return 0
    take = min(sample_n, n)
    sample = hf_split.select(list(range(take)))
    labels = [int(x) for x in sample[label_key]]
    return -1 if labels and min(labels) == 1 else 0

def get_dataset(
    dataset_name: str,
    processor,
    *,
    data_dir: str = "./downstream_datasets",
    imagenet100_path: Optional[str] = None,
    officehome_source: Optional[str] = None,
    officehome_target: Optional[str] = None,
    domainnet_domain: str = "clipart",
    train_transform=None,
    test_transform=None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    """Centralized dataset loader for classification tasks."""
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    if train_transform is None or test_transform is None:
        train_transform, test_transform = _make_transforms(processor)
    logger = logging.getLogger(__name__)

    if dataset_name == 'cifar100':
        train_set = datasets.CIFAR100(root=str(data_dir_path), train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR100(root=str(data_dir_path), train=False, download=True, transform=test_transform)
        num_classes = 100
    elif dataset_name == 'caltech256':
        from datasets import load_dataset
        ds: Any = load_dataset("ilee0022/Caltech-256", cache_dir=str(data_dir_path / "hf_cache"))
        train_split = ds["train"]
        test_split = ds["test"]
        # Caltech-256 classes are 1-257 (256 objects + 1 clutter)
        # We shift them to 0-256 for standard classification
        train_set = _wrap_hf_dataset(train_split, train_transform, label_offset=-1)
        test_set = _wrap_hf_dataset(test_split, test_transform, label_offset=-1)
        num_classes = 257
    elif dataset_name == 'cub200':
        from datasets import load_dataset
        hf_dataset: Any = load_dataset("bentrevett/caltech-ucsd-birds-200-2011", cache_dir=str(data_dir_path / "hf_cache"))
        train_split = hf_dataset["train"]
        test_split = hf_dataset["test"]
        label_offset = _infer_label_offset(train_split, label_key="label")
        train_set = _wrap_hf_dataset(train_split, train_transform, label_offset=label_offset)
        test_set = _wrap_hf_dataset(test_split, test_transform, label_offset=label_offset)
        num_classes = 200
    elif dataset_name == 'imagenet100':
        root = Path(imagenet100_path) if imagenet100_path else None
        if root and root.exists():
            train_root = root / "train"
            val_root = root / "val" if (root / "val").exists() else root / "test"
            train_set = datasets.ImageFolder(root=str(train_root), transform=train_transform)
            test_set = datasets.ImageFolder(root=str(val_root), transform=test_transform)
            num_classes = len(train_set.classes)
        else:
            from datasets import load_dataset, load_from_disk
            saved_dir = data_dir_path / "imagenet100"
            if saved_dir.exists():
                ds: Any = load_from_disk(str(saved_dir))
            else:
                ds = load_dataset("clane9/imagenet-100", cache_dir=str(data_dir_path / "hf_cache"))
                ds.save_to_disk(str(saved_dir))
            train_split = ds["train"]
            val_split = ds["validation"] if "validation" in ds else ds["val"]
            label_offset = _infer_label_offset(train_split, label_key="label")
            train_set = _wrap_hf_dataset(train_split, train_transform, label_offset=label_offset)
            test_set = _wrap_hf_dataset(val_split, test_transform, label_offset=label_offset)
            num_classes = 100
    elif dataset_name == 'officehome':
        from datasets import load_dataset
        source = (officehome_source or "product").lower()
        target = (officehome_target or "realworld").lower()
        dataset_domain_map = {"art": "art", "clipart": "clipart", "product": "product", "realworld": "real world"}
        ds: Any = load_dataset("flwrlabs/office-home", cache_dir=str(data_dir_path / "hf_cache"))
        split = ds["train"]
        domain_col = [str(d).lower() for d in split["domain"]]
        src_idx = [i for i, d in enumerate(domain_col) if d == dataset_domain_map.get(source, source)]
        tgt_idx = [i for i, d in enumerate(domain_col) if d == dataset_domain_map.get(target, target)]
        train_set = _wrap_hf_dataset(split.select(src_idx), train_transform)
        test_set = _wrap_hf_dataset(split.select(tgt_idx), test_transform)
        num_classes = 65
    elif dataset_name == 'domainnet_clipart':
        from datasets import load_dataset
        try:
            ds: Any = load_dataset("wltjr1007/DomainNet", cache_dir=str(data_dir_path / "hf_cache"))
        except Exception:
            ds = load_dataset("wltjr1007/DomainNet", cache_dir=str(data_dir_path / "hf_cache"), download_mode="force_redownload")
        domain_name = domainnet_domain.lower()
        train_split = ds["train"]
        test_split = ds["test"]
        train_features = cast(Any, train_split.features)
        domain_feat = train_features.get("domain") if train_features is not None else None
        domain_names = list(getattr(domain_feat, "names", [])) if domain_feat is not None else []
        domain_id = domain_names.index(domain_name) if domain_name in domain_names else None
        train_idx = [i for i, d in enumerate(ds["train"]["domain"]) if int(d) == domain_id]
        test_idx = [i for i, d in enumerate(ds["test"]["domain"]) if int(d) == domain_id]
        train_set = _wrap_hf_dataset(train_split.select(train_idx), train_transform)
        test_set = _wrap_hf_dataset(test_split.select(test_idx), test_transform)
        num_classes = 345
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_set, test_set, num_classes
