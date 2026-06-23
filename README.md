# MobileSAM Distillation

MobileSAM image-encoder distillation, checkpoint aggregation, and COCO prompt benchmarking.

The project keeps reusable training and evaluation code in an installable Python package. Datasets, teacher embeddings, large checkpoints, logs, and experiment outputs are external artifacts mounted at runtime.

## Project Structure

```text
src/mobilesam_distill/
  data/          dataset loaders and COCO sample preparation
  teacher/       SAM teacher feature export
  training/      image-encoder distillation loop
  models/        student model builders and checkpoint aggregation
  evaluation/    COCO prompt benchmark
  cli/           smoke-test entrypoints
configs/         example runtime settings
scripts/         container and local helper commands
requirements/    dependency groups
weights/         small committed reference weights
```

The upstream MobileSAM implementation is installed during image build from a pinned git commit:

```text
https://github.com/ChaoningZhang/MobileSAM.git
01ea8d0f5590082f0c1ceb0a3e2272593f20154b
```

Set `MOBILE_SAM_REPO` during build to use another mirror.

Inside the container, the third-party source is cloned to `/opt/MobileSAM`, installed editable, and exposed as the `mobile_sam` Python package. It is intentionally not stored as a submodule in this repository.

## Artifact Layout

Use this runtime layout for local runs and container runs:

```text
/artifacts/data          read-only source datasets
/artifacts/features      SAM teacher embeddings
/artifacts/checkpoints   large external checkpoints
/artifacts/outputs       training runs, benchmark outputs, logs
/artifacts/cache         package/model/tool caches
```

Source datasets should be treated as read-only. Teacher features are written under `FEATURE_ROOT`, not next to images.

Small final MobileSAM-compatible checkpoints may be committed under `weights/distilled/`. Do not commit SAM teacher checkpoints, teacher embeddings, raw training checkpoints, benchmark outputs, overlays, or logs.

## Container Setup

Build the image:

```bash
bash scripts/container/build.sh
```

Run an import and model-construction smoke check:

```bash
bash scripts/container/smoke_import.sh
```

Equivalent manual run:

```bash
docker run --rm --gpus all \
  -v "$PWD/artifacts:/artifacts" \
  mobilesam-distill:latest \
  mobilesam-smoke-import
```

## Local Editable Setup

Use this only when not running in the container:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements/train.txt
pip install -e .
```

Install MobileSAM separately from the pinned commit, or run inside the container where it is already installed.

## Validation Samples

Prepare a 10-image COCO val2017 sample from an existing COCO root:

```bash
ARTIFACT_ROOT="$PWD/artifacts" COCO_ROOT=/path/to/coco \
  bash scripts/local/prepare_coco10.sh
```

Or download COCO temporarily and keep only the sample:

```bash
ARTIFACT_ROOT="$PWD/artifacts" bash scripts/local/prepare_coco10.sh
```

Output:

```text
artifacts/data/coco10/val2017/*.jpg
artifacts/data/coco10/annotations/instances_val2017.json
```

Prepare a 5-image validation sample:

```bash
ARTIFACT_ROOT="$PWD/artifacts" COCO_ROOT=/path/to/coco \
  bash scripts/local/prepare_val5.sh
```

Output:

```text
artifacts/data/coco_val5/val2017/*.jpg
artifacts/data/coco_val5/annotations/instances_val2017.json
```

## Distillation Pipeline

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
