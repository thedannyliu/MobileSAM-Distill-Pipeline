# Weights

Small final checkpoints that are useful for inference smoke tests may be committed here.

Committed:

- `distilled/mobile_sam.pt`: upstream MobileSAM checkpoint used as the prompt/mask shell.

Do not commit:

- SAM1 teacher checkpoints, such as `sam_vit_h_4b8939.pth`
- teacher embeddings (`*.npy`)
- raw resumable training checkpoints (`last.pt`, `best_val.pt`, `iter_*.pt`)
- benchmark outputs, overlays, or logs

Use `/artifacts/checkpoints` or another mounted checkpoint directory for large external weights.
