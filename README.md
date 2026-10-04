# MobileSAM distillation

Train a compact SAM image encoder from teacher features, assemble a
MobileSAM-compatible checkpoint, and evaluate point-prompt segmentation on COCO.
The package separates data preparation, teacher export, training and evaluation.

## Install

For a CUDA environment with a compatible PyTorch/torchvision pair:

```bash
python -m pip install -e .
python -m pip install 'git+https://github.com/ChaoningZhang/MobileSAM.git@01ea8d0f5590082f0c1ceb0a3e2272593f20154b'
mobilesam-smoke-import
```

Or build the supplied container with `bash scripts/container/build.sh`, then
run `bash scripts/container/smoke_import.sh`. It installs the same pinned
MobileSAM source under `/opt/MobileSAM`.

## Workflow

| Step | Entry point |
| --- | --- |
| Prepare a reproducible COCO subset | `mobilesam-prepare-coco` |
| Export SAM teacher embeddings | `mobilesam-export-teacher` |
| Train the student encoder | `mobilesam-train` or `torchrun -m mobilesam_distill.training.distill` |
| Assemble encoder + prompt/mask heads | `mobilesam-aggregate` |
| Evaluate a checkpoint | `mobilesam-benchmark-coco` |

See the [training and evaluation recipe](docs/training.md) for a complete run.
All commands accept `--help`.

Prepare any subset size with one command:

```bash
mobilesam-prepare-coco --coco_root /path/to/coco \
  --num_images 5 --seed 1234 --output_root artifacts/data/coco_val5
```

COCO must contain `val2017/` and `annotations/instances_val2017.json`.
Use `--download` instead of `--coco_root` only when you intend to download
the full validation archive before sampling. A five-image smoke checks the
pipeline; it is not a dataset-level accuracy result.

## Layout and artifacts

- `src/mobilesam_distill/`: reusable data, teacher, training, model and evaluation code.
- `configs/`: runtime examples; `scripts/`: container and checkpoint helpers.
- `weights/distilled/`: reference MobileSAM-compatible weights.
- `artifacts/`: ignored datasets, features, checkpoints and generated outputs.

Keep source datasets read-only. Write teacher embeddings separately from images,
and keep large checkpoints and logs outside git. Benchmark reports include
per-image metrics and overlays; compare runs only with matching prompt protocols,
checkpoints, hardware and timing scopes.

The older `mobilesam-prepare-coco10` command remains an alias. The redundant
`prepare_coco10.sh` and `prepare_val5.sh` wrappers have been replaced by the
parameterized command above. Unreferenced generated `vis/` images were removed
from the source tree; prior outputs remain in git history. Upstream models and
weights retain their terms.
