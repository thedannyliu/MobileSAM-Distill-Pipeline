import argparse
import json
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def parse_args():
    parser = argparse.ArgumentParser("Prepare a fixed-size COCO val sample")
    parser.add_argument("--coco_root", type=str, default="", help="Existing COCO root with val2017 and annotations")
    parser.add_argument("--output_root", type=str, default="sample_data/coco10")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--download", action="store_true", help="Download COCO val2017 if --coco_root is not provided")
    parser.add_argument("--keep_download", action="store_true", help="Keep the temporary full COCO download")
    return parser.parse_args()


def download_and_extract(url, work_dir):
    archive = work_dir / Path(url).name
    if not archive.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(work_dir)


def resolve_coco_root(args):
    if args.coco_root:
        return Path(args.coco_root).resolve(), None
    if not args.download:
        raise ValueError("Pass --coco_root or use --download.")
    work_dir = Path(tempfile.mkdtemp(prefix="mobilesam_coco_"))
    download_and_extract(COCO_VAL_URL, work_dir)
    download_and_extract(COCO_ANN_URL, work_dir)
    return work_dir, work_dir


def main():
    args = parse_args()
    coco_root, temp = resolve_coco_root(args)
    try:
        image_dir = coco_root / "val2017"
        ann_path = coco_root / "annotations" / "instances_val2017.json"
        if not image_dir.is_dir() or not ann_path.is_file():
            raise FileNotFoundError(f"COCO val2017 layout is missing under {coco_root}")

        annotations = json.loads(ann_path.read_text())
        images = sorted(annotations["images"], key=lambda item: item["id"])
        rng = random.Random(args.seed)
        selected = sorted(rng.sample(images, min(args.num_images, len(images))), key=lambda item: item["id"])
        selected_ids = {item["id"] for item in selected}

        output_root = Path(args.output_root)
        output_images = output_root / "val2017"
        output_ann = output_root / "annotations"
        output_images.mkdir(parents=True, exist_ok=True)
        output_ann.mkdir(parents=True, exist_ok=True)

        for image in selected:
            shutil.copy2(image_dir / image["file_name"], output_images / image["file_name"])

        subset = dict(annotations)
        subset["images"] = selected
        subset["annotations"] = [ann for ann in annotations["annotations"] if ann["image_id"] in selected_ids]
        subset["info"] = {
            **annotations.get("info", {}),
            "description": f"COCO val2017 {len(selected)} image sample for MobileSAM smoke tests",
            "seed": args.seed,
        }
        (output_ann / "instances_val2017.json").write_text(json.dumps(subset, indent=2) + "\n")
        print(f"Wrote {len(selected)} image COCO sample to {output_root}")
    finally:
        if temp is not None and args.keep_download:
            print(f"Keeping full COCO download at {temp}")
        elif temp is not None:
            shutil.rmtree(temp)


if __name__ == "__main__":
    main()
