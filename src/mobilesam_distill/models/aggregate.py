import argparse
import os

import torch

from mobilesam_distill.models.student import build_mobilesam_shell, load_image_encoder_checkpoint


def parse_option():
    parser = argparse.ArgumentParser("Aggregate a distilled image encoder into a MobileSAM checkpoint")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--student_arch", type=str, default="tinyvit", choices=["tinyvit", "repvit_m0_9"])
    parser.add_argument("--mobile_sam_type", type=str, default="vit_t", help="Kept for compatibility; only vit_t decoder is supported.")
    parser.add_argument("--mobile_sam_ckpt", type=str, default="weights/distilled/mobile_sam.pt")
    parser.add_argument("--save_model_path", type=str, default="/artifacts/outputs")
    parser.add_argument("--save_model_name", type=str, default="our_retrained_mobilesam.pth")
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_option()
    print("Building MobileSAM shell ...")
    mobile_sam = build_mobilesam_shell(args.student_arch, mobile_sam_ckpt=args.mobile_sam_ckpt)
    missing, unexpected = load_image_encoder_checkpoint(mobile_sam, args.ckpt, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Image encoder load mismatch. missing={missing}, unexpected={unexpected}")

    os.makedirs(args.save_model_path, exist_ok=True)
    save_path = os.path.join(args.save_model_path, args.save_model_name)
    torch.save(mobile_sam.state_dict(), save_path)
    print(f"Completed. Aggregated model saved as {save_path}")


if __name__ == "__main__":
    main(parse_option())
