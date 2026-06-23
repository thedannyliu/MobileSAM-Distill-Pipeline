import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry
from tqdm import tqdm


def parse_option():
    parser = argparse.ArgumentParser("Export SAM teacher image embeddings")

    parser.add_argument("--dataset_path", type=str, default="/artifacts/data/SA-1B-MobileSAM", help="root path of dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="image directory relative to dataset path")
    parser.add_argument(
        "--feature_root",
        type=str,
        default="",
        help="root for exported .npy features; defaults to writing beside images",
    )
    parser.add_argument("--manifest_path", type=str, default="", help="optional teacher manifest path")

    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--sam_type", type=str, default="vit_h")
    parser.add_argument("--sam_ckpt", type=str, default="/artifacts/checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--max_images", type=int, default=-1, help="limit images for smoke runs")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing .npy features")

    return parser.parse_args()


def feature_path_for_image(dataset_path, feature_root, image_path):
    if not feature_root:
        return Path(image_path).with_suffix(".npy")
    rel_path = Path(image_path).relative_to(Path(dataset_path).resolve())
    return Path(feature_root) / rel_path.with_suffix(".npy")


def main(args=None):
    if args is None:
        args = parse_option()
    device = args.device

    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    dataset_path = Path(args.dataset_path).resolve()
    feature_root = Path(args.feature_root).resolve() if args.feature_root else None
    test_image_dir = dataset_path / args.dataset_dir
    test_image_paths = [
        test_image_dir / img_name
        for img_name in sorted(os.listdir(test_image_dir))
        if img_name.lower().endswith((".jpg", ".jpeg"))
    ]
    if args.max_images > 0:
        test_image_paths = test_image_paths[:args.max_images]

    manifest_rows = []
    n = len(test_image_paths)
    for i, test_image_path in enumerate(tqdm(test_image_paths)):
        feature_path = feature_path_for_image(dataset_path, feature_root, test_image_path)
        if feature_path.exists() and not args.overwrite:
            manifest_rows.append({"image": str(test_image_path), "feature": str(feature_path), "exported": False})
            continue

        print(i, "/", n)
        test_image = cv2.imread(str(test_image_path))
        if test_image is None:
            raise FileNotFoundError(f"Could not read image: {test_image_path}")
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            predictor.set_image(test_image)
            feature = predictor.features
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=feature_path.name + ".",
            suffix=".tmp.npy",
            dir=feature_path.parent,
        )
        os.close(fd)
        try:
            np.save(tmp_path, feature.cpu().numpy())
            os.replace(tmp_path, feature_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        manifest_rows.append({"image": str(test_image_path), "feature": str(feature_path), "exported": True})

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        missing = [row for row in manifest_rows if not Path(row["feature"]).exists()]
        manifest = {
            "dataset_path": str(dataset_path),
            "dataset_dir": args.dataset_dir,
            "feature_root": str(feature_root) if feature_root else "",
            "sam_type": args.sam_type,
            "sam_ckpt": args.sam_ckpt,
            "num_images": len(manifest_rows),
            "num_missing_teacher_features": len(missing),
            "features": manifest_rows,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main(parse_option())
