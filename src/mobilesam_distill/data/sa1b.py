import cv2
import numpy as np
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from mobile_sam.utils.transforms import ResizeLongestSide


class sa1b_dataset(Dataset):
    def __init__(self, root_path, img_dirs, transformer, max_num=None, feature_root=None):
        self.root_path = os.path.abspath(root_path)
        self.img_dirs = img_dirs
        self.transformer = transformer
        self.max_num = max_num
        self.feature_root = os.path.abspath(feature_root) if feature_root else None
        self.img_paths = []
        for img_dir in img_dirs:
            img_root = os.path.join(self.root_path, img_dir)
            img_names = sorted(os.listdir(img_root))
            self.img_paths += [
                os.path.join(img_root, img_name)
                for img_name in img_names
                if img_name.lower().endswith((".jpg", ".jpeg"))
            ]

    def __len__(self):
        if not self.max_num:
            return len(self.img_paths)
        return min(self.max_num, len(self.img_paths))

    def feature_path_for_image(self, image_path):
        if not self.feature_root:
            return os.path.splitext(image_path)[0] + ".npy"
        rel_path = os.path.relpath(image_path, self.root_path)
        return os.path.join(self.feature_root, os.path.splitext(rel_path)[0] + ".npy")

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        if img is None:
            raise FileNotFoundError(f"Could not read image: {self.img_paths[index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transformer:
            img = self.transformer(img)

        feat_path = self.feature_path_for_image(self.img_paths[index])
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Teacher feature is missing for {self.img_paths[index]}. "
                f"Expected feature at {feat_path}. Run teacher export first."
            )
        feat = np.load(feat_path).squeeze()

        return img, feat, os.path.splitext(self.img_paths[index])[0] + ".json"


def transform(x, img_size=1024):
    """Normalize pixel values and pad to a square input."""
    resize = ResizeLongestSide(img_size)
    x = resize.apply_image(x)
    x = torch.as_tensor(x)
    x = x.permute(2, 0, 1).contiguous()

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std

    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def get_sa1b_dataloaders(
    transformer,
    root_path,
    train_dirs,
    val_dirs,
    batch_size=4,
    num_workers=4,
    val_max_num=1000,
    feature_root=None,
):
    train_set = sa1b_dataset(root_path, train_dirs, transformer, feature_root=feature_root)
    val_set = sa1b_dataset(root_path, val_dirs, transformer, val_max_num, feature_root=feature_root)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, val_loader


if __name__ == "__main__":
    root_path = os.environ.get("DATA_ROOT", "/artifacts/data/SA-1B-MobileSAM")
    feature_root = os.environ.get("FEATURE_ROOT")
    train_dirs = ["sa_00000" + str(i) for i in range(10)]
    val_dirs = ["sa_000010"]
    transformer = transform
    train_loader, val_loader = get_sa1b_dataloaders(
        transformer,
        root_path,
        train_dirs,
        val_dirs,
        feature_root=feature_root,
    )
    print(len(val_loader))
