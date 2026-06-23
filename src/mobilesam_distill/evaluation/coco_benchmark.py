import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from mobile_sam import SamPredictor, sam_model_registry
from mobilesam_distill.models.student import build_mobilesam_shell, load_image_encoder_checkpoint, torch_load


def parse_args():
    parser = argparse.ArgumentParser("COCO val point-prompt benchmark for MobileSAM variants")
    parser.add_argument("--coco_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="", help="Full aggregated SAM checkpoint")
    parser.add_argument("--image_encoder_ckpt", type=str, default="", help="Image encoder-only checkpoint")
    parser.add_argument("--mobile_sam_ckpt", type=str, default="weights/distilled/mobile_sam.pt")
    parser.add_argument("--student_arch", type=str, default="tinyvit", choices=["tinyvit", "repvit_m0_9", "sam_vit_b"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--overlay_count", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup_first_image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt_type", type=str, default="point", choices=["point", "box"])
    parser.add_argument("--box_expand", type=float, default=1.0)
    return parser.parse_args()


def synchronize(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def build_model(args):
    if args.student_arch == "sam_vit_b":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for sam_vit_b")
        model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
        model.to(args.device)
        model.eval()
        return model

    if args.checkpoint:
        model = build_mobilesam_shell(args.student_arch, mobile_sam_ckpt=None)
        state_dict = torch_load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    else:
        model = build_mobilesam_shell(args.student_arch, mobile_sam_ckpt=args.mobile_sam_ckpt)
        if args.image_encoder_ckpt:
            load_image_encoder_checkpoint(model, args.image_encoder_ckpt, strict=True)
    model.to(args.device)
    model.eval()
    return model


def choose_annotation(coco, image_id):
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    anns = [ann for ann in anns if ann.get("area", 0) > 0]
    if not anns:
        return None
    return max(anns, key=lambda ann: ann.get("area", 0))


def point_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x = int(round(float(xs.mean())))
    y = int(round(float(ys.mean())))
    if mask[y, x] == 0:
        mid = len(xs) // 2
        x = int(xs[mid])
        y = int(ys[mid])
    return np.array([[x, y]], dtype=np.float32)


def box_from_mask(mask, expand=1.0):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max())
    y1 = float(ys.max())
    if expand != 1.0:
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        w = max(1.0, (x1 - x0 + 1.0) * expand)
        h = max(1.0, (y1 - y0 + 1.0) * expand)
        x0 = cx - w * 0.5
        x1 = cx + w * 0.5
        y0 = cy - h * 0.5
        y1 = cy + h * 0.5
    height, width = mask.shape[:2]
    x0 = max(0.0, min(float(width - 1), x0))
    x1 = max(0.0, min(float(width - 1), x1))
    y0 = max(0.0, min(float(height - 1), y0))
    y1 = max(0.0, min(float(height - 1), y1))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    true_negative = np.logical_and(~pred, ~gt).sum()
    total = pred.size
    iou = intersection / union if union else 1.0
    dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) else 1.0
    precision = intersection / pred_sum if pred_sum else 0.0
    recall = intersection / gt_sum if gt_sum else 0.0
    accuracy = (intersection + true_negative) / total if total else 0.0
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(accuracy),
    }


def save_overlay(path, image_rgb, gt_mask, pred_mask, point=None, box=None):
    overlay = image_rgb.copy()
    gt_color = np.array([0, 255, 0], dtype=np.uint8)
    pred_color = np.array([255, 0, 0], dtype=np.uint8)
    both_color = np.array([255, 255, 0], dtype=np.uint8)
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    overlay[gt] = (0.55 * overlay[gt] + 0.45 * gt_color).astype(np.uint8)
    overlay[pred] = (0.55 * overlay[pred] + 0.45 * pred_color).astype(np.uint8)
    both = np.logical_and(gt, pred)
    overlay[both] = (0.45 * overlay[both] + 0.55 * both_color).astype(np.uint8)
    if point is not None:
        x, y = point[0].astype(int).tolist()
        cv2.circle(overlay, (x, y), 6, (255, 255, 255), -1)
        cv2.circle(overlay, (x, y), 6, (0, 0, 0), 2)
    if box is not None:
        x0, y0, x1, y1 = box.astype(int).tolist()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), 1)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the benchmark job.")

    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    coco_root = Path(args.coco_root)
    image_dir = coco_root / "val2017"
    ann_path = coco_root / "annotations" / "instances_val2017.json"
    if not image_dir.is_dir() or not ann_path.is_file():
        raise FileNotFoundError(f"COCO val2017 is missing under {coco_root}")

    model = build_model(args)
    predictor = SamPredictor(model)
    coco = COCO(str(ann_path))
    image_ids = sorted(coco.getImgIds())
    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]

    rows = []
    total_time = 0.0
    encoder_time = 0.0
    decoder_time = 0.0
    overlay_written = 0
    warmed_up = False

    for image_id in image_ids:
        ann = choose_annotation(coco, image_id)
        if ann is None:
            continue
        info = coco.loadImgs([image_id])[0]
        image_path = image_dir / info["file_name"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = coco.annToMask(ann).astype(bool)
        point = point_from_mask(gt_mask)
        box = box_from_mask(gt_mask, expand=args.box_expand)
        if point is None or box is None:
            continue

        labels = np.array([1], dtype=np.int32)
        if args.warmup_first_image and not warmed_up:
            predictor.set_image(image_rgb)
            if args.prompt_type == "box":
                predictor.predict(box=box, multimask_output=True)
            else:
                predictor.predict(point_coords=point, point_labels=labels, multimask_output=True)
            synchronize(args.device)
            warmed_up = True

        synchronize(args.device)
        start_total = time.perf_counter()
        start_encoder = time.perf_counter()
        predictor.set_image(image_rgb)
        synchronize(args.device)
        end_encoder = time.perf_counter()
        if args.prompt_type == "box":
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        else:
            masks, scores, _ = predictor.predict(point_coords=point, point_labels=labels, multimask_output=True)
        synchronize(args.device)
        end_total = time.perf_counter()

        best_idx = int(np.argmax(scores))
        pred_mask = masks[best_idx].astype(bool)
        metrics = compute_metrics(pred_mask, gt_mask)
        enc_time = end_encoder - start_encoder
        dec_time = end_total - end_encoder
        elapsed = end_total - start_total
        total_time += elapsed
        encoder_time += enc_time
        decoder_time += dec_time
        row = {
            "image_id": image_id,
            "file_name": info["file_name"],
            "annotation_id": ann["id"],
            "category_id": ann["category_id"],
            "score": float(scores[best_idx]),
            "total_time_sec": elapsed,
            "encoder_time_sec": enc_time,
            "decoder_time_sec": dec_time,
            "prompt_type": args.prompt_type,
            "box_expand": args.box_expand,
            "box_x0": float(box[0]),
            "box_y0": float(box[1]),
            "box_x1": float(box[2]),
            "box_y1": float(box[3]),
            **metrics,
        }
        rows.append(row)

        if overlay_written < args.overlay_count:
            save_overlay(
                overlay_dir / f"{overlay_written:02d}_{image_id}.png",
                image_rgb,
                gt_mask,
                pred_mask,
                point=point if args.prompt_type == "point" else None,
                box=box if args.prompt_type == "box" else None,
            )
            overlay_written += 1

    if not rows:
        raise RuntimeError("No valid COCO examples were evaluated.")

    summary = {
        "run_name": args.run_name or Path(args.checkpoint or args.image_encoder_ckpt or args.mobile_sam_ckpt).stem,
        "student_arch": args.student_arch,
        "num_images": len(rows),
        "fps_total": len(rows) / total_time if total_time else 0.0,
        "fps_encoder": len(rows) / encoder_time if encoder_time else 0.0,
        "fps_decoder": len(rows) / decoder_time if decoder_time else 0.0,
        "mean_total_time_sec": total_time / len(rows),
        "mean_encoder_time_sec": encoder_time / len(rows),
        "mean_decoder_time_sec": decoder_time / len(rows),
        "mIoU": float(np.mean([row["iou"] for row in rows])),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_recall": float(np.mean([row["recall"] for row in rows])),
        "mean_pixel_accuracy": float(np.mean([row["pixel_accuracy"] for row in rows])),
        "iou_at_50": float(np.mean([row["iou"] >= 0.50 for row in rows])),
        "iou_at_75": float(np.mean([row["iou"] >= 0.75 for row in rows])),
        "overlay_dir": str(overlay_dir),
        "warmup_first_image": args.warmup_first_image,
        "prompt_type": args.prompt_type,
        "box_expand": args.box_expand,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    with open(output_dir / "per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
