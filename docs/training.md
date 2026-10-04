# Distillation workflow

Set common paths:

```bash
export ARTIFACT_ROOT="$PWD/artifacts"
export DATA_ROOT="${ARTIFACT_ROOT}/data/SA-1B-MobileSAM"
export FEATURE_ROOT="${ARTIFACT_ROOT}/features/SA-1B-MobileSAM"
export CHECKPOINT_ROOT="${ARTIFACT_ROOT}/checkpoints"
export OUTPUT_ROOT="${ARTIFACT_ROOT}/outputs"
```

Prepare checkpoint mounts:

```bash
ARTIFACT_ROOT="$PWD/artifacts" DOWNLOAD_SAM_TEACHER=1 \
  bash scripts/local/prepare_checkpoints.sh
```

Export SAM teacher features:

```bash
mobilesam-export-teacher \
  --dataset_path "${DATA_ROOT}" \
  --dataset_dir images/train \
  --feature_root "${FEATURE_ROOT}" \
  --sam_ckpt "${CHECKPOINT_ROOT}/sam_vit_h_4b8939.pth" \
  --manifest_path "${FEATURE_ROOT}/teacher_images_train_manifest.json"
```

Train from existing teacher features:

```bash
torchrun --standalone --nproc_per_node=1 -m mobilesam_distill.training.distill \
  --dataset_path "${DATA_ROOT}" \
  --feature_root "${FEATURE_ROOT}" \
  --train_dirs images/train \
  --val_dirs images/val \
  --student_arch tinyvit \
  --epochs 1 \
  --batch_size 1 \
  --max_train_samples 2 \
  --eval_nums 2 \
  --root_path "${OUTPUT_ROOT}" \
  --work_dir smoke
```

Aggregate a trained image encoder:

```bash
mobilesam-aggregate \
  --ckpt "${OUTPUT_ROOT}/smoke/ckpt/iter_final.pth" \
  --mobile_sam_ckpt weights/distilled/mobile_sam.pt \
  --save_model_path "${OUTPUT_ROOT}" \
  --save_model_name mobilesam_smoke_aggregated.pth
```

Evaluate the aggregated checkpoint on the 5-image validation sample:

```bash
CHECKPOINT="${OUTPUT_ROOT}/mobilesam_smoke_aggregated.pth" \
  ARTIFACT_ROOT="$PWD/artifacts" \
  bash scripts/local/benchmark_coco.sh
```

Benchmark outputs:

```text
artifacts/outputs/bench_coco_val5/summary.json
artifacts/outputs/bench_coco_val5/per_image.csv
artifacts/outputs/bench_coco_val5/overlays/*.png
```

The summary reports latency, FPS, mIoU, Dice, precision, recall, pixel accuracy, and IoU thresholds. Use `per_image.csv` and overlays to inspect failures.

Benchmark directly on another COCO-format validation set:

```bash
mobilesam-benchmark-coco \
  --coco_root "${ARTIFACT_ROOT}/data/coco_val5" \
  --checkpoint "${OUTPUT_ROOT}/mobilesam_smoke_aggregated.pth" \
  --output_dir "${OUTPUT_ROOT}/bench_coco_val5" \
  --max_images 5
```

Run all commands from the repository root. Prepare `artifacts/data/coco_val5`
with `mobilesam-prepare-coco --num_images 5 --coco_root /path/to/coco
--output_root artifacts/data/coco_val5` before the evaluation example.
