def main():
    import torch
    from mobile_sam import sam_model_registry
    from mobilesam_distill.models import build_student_image_encoder

    tinyvit = build_student_image_encoder("tinyvit")
    repvit = build_student_image_encoder("repvit_m0_9")
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("mobile_sam_path", getattr(__import__("mobile_sam"), "__file__", ""))
    print("mobile_sam_models", sorted(sam_model_registry.keys()))
    print("tinyvit_params", sum(p.numel() for p in tinyvit.parameters()))
    print("repvit_m0_9_params", sum(p.numel() for p in repvit.parameters()))


if __name__ == "__main__":
    main()
