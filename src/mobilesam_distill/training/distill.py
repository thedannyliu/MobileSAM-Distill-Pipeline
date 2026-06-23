import argparse
import os
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mobilesam_distill.data.sa1b import sa1b_dataset, transform
from mobilesam_distill.models.student import build_student_image_encoder

STOP_REQUESTED = False


def _handle_stop_signal(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; checkpointing after the current batch.", flush=True)


signal.signal(signal.SIGTERM, _handle_stop_signal)
signal.signal(signal.SIGINT, _handle_stop_signal)


def parse_option():
    parser = argparse.ArgumentParser("MobileSAM image encoder distillation")

    parser.add_argument("--dataset_path", type=str, default="/artifacts/data/SA-1B-MobileSAM")
    parser.add_argument(
        "--feature_root",
        type=str,
        default="",
        help="Root for teacher .npy features. Defaults to adjacent files beside images.",
    )
    parser.add_argument("--train_dirs", type=str, default=",".join(["sa_" + str(i).zfill(6) for i in range(20)]))
    parser.add_argument("--val_dirs", type=str, default="sa_000020")
    parser.add_argument("--max_train_samples", type=int, default=-1)

    parser.add_argument("--student_arch", type=str, default="tinyvit", choices=["tinyvit", "repvit_m0_9"])
    parser.add_argument("--loss_mse_weight", type=float, default=1.0)
    parser.add_argument("--loss_cosine_weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", type=bool, default=True)
    parser.add_argument("--benchmark", type=bool, default=False)

    parser.add_argument("--optim", type=str, default="sgd", choices=["adam", "sgd", "adamw"])
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--print_iters", type=int, default=200)
    parser.add_argument("--eval_nums", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=0)

    parser.add_argument("--root_path", type=str, default="/artifacts/outputs")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--save_dir", type=str, default="ckpt")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--save_iters", type=int, default=50000)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--auto_resume", action="store_true")

    return parser.parse_args()


def parse_dirs(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def model_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def load_model_state_dict(model, state_dict):
    if hasattr(model, "module"):
        return model.module.load_state_dict(state_dict)
    return model.load_state_dict(state_dict)


def get_optimizer(args, model):
    if args.optim == "adam":
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optim == "sgd":
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optim == "adamw":
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    raise NotImplementedError(args.optim)


def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)


def mse_distill_loss(pred_feats, target_feats):
    return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()


def cosine_distill_loss(pred_feats, target_feats):
    pred = pred_feats.flatten(1)
    target = target_feats.flatten(1)
    return 1.0 - F.cosine_similarity(pred, target, dim=1).mean()


def compute_losses(args, pred_feats, target_feats):
    mse_loss = mse_distill_loss(pred_feats, target_feats)
    cosine_loss = cosine_distill_loss(pred_feats, target_feats)
    total_loss = args.loss_mse_weight * mse_loss + args.loss_cosine_weight * cosine_loss
    return total_loss, {"mse_loss": mse_loss, "cosine_loss": cosine_loss, "total_loss": total_loss}


def reduce_mean(tensor):
    if not is_dist_avail_and_initialized():
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced


def reduce_loss_dict(loss_dict):
    return {key: reduce_mean(value).item() for key, value in loss_dict.items()}


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu().to(torch.uint8))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([cuda_state.detach().cpu().to(torch.uint8) for cuda_state in state["cuda"]])


def save_checkpoint(path, args, model, optimizer, scheduler, scaler, epoch, batch_idx, global_step, best_val_loss, epoch_complete):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model": model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "student_arch": args.student_arch,
        "loss_mse_weight": args.loss_mse_weight,
        "loss_cosine_weight": args.loss_cosine_weight,
        "epoch_complete": epoch_complete,
        "rng_state": get_rng_state(),
    }
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def resolve_resume_path(args):
    if args.resume:
        return args.resume
    if args.auto_resume:
        candidate = os.path.join(args.root_path, args.work_dir, args.save_dir, "last.pt")
        if os.path.exists(candidate):
            return candidate
    return ""


def load_checkpoint(path, args, model, optimizer, scheduler, scaler, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    load_model_state_dict(model, checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    set_rng_state(checkpoint.get("rng_state"))
    if checkpoint.get("student_arch") and checkpoint["student_arch"] != args.student_arch:
        raise ValueError(f"Checkpoint arch {checkpoint['student_arch']} does not match --student_arch {args.student_arch}")
    if checkpoint.get("epoch_complete", False):
        start_epoch = int(checkpoint["epoch"]) + 1
        resume_batch_idx = -1
    else:
        start_epoch = int(checkpoint["epoch"]) + 1
        resume_batch_idx = -1
    return {
        "start_epoch": start_epoch,
        "resume_batch_idx": resume_batch_idx,
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
    }


def evaluate(args, model, val_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, target_feats, _ in val_loader:
            imgs = imgs.to(args.device, non_blocking=True)
            target_feats = target_feats.to(args.device, non_blocking=True)
            pred_feats = model(imgs)
            loss, _ = compute_losses(args, pred_feats, target_feats)
            total_loss += loss.item()
    avg_loss = torch.tensor(total_loss / max(1, len(val_loader)), device=args.device)
    return reduce_mean(avg_loss).item()


def main(args=None):
    if args is None:
        args = parse_option()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")

    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda", 0)
    args.device = device

    save_root = os.path.join(args.root_path, args.work_dir, args.save_dir)
    log_root = os.path.join(args.root_path, args.work_dir, args.log_dir)
    if is_main_process():
        os.makedirs(save_root, exist_ok=True)
        os.makedirs(log_root, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = args.deterministic
    cudnn.benchmark = args.benchmark

    train_dirs = parse_dirs(args.train_dirs)
    val_dirs = parse_dirs(args.val_dirs)
    train_max = None if args.max_train_samples <= 0 else args.max_train_samples
    feature_root = args.feature_root or None
    train_dataset = sa1b_dataset(args.dataset_path, train_dirs, transform, train_max, feature_root=feature_root)
    val_dataset = sa1b_dataset(args.dataset_path, val_dirs, transform, args.eval_nums, feature_root=feature_root)
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    per_rank_batch_size = max(1, args.batch_size // get_world_size())
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(val_dataset, batch_size=per_rank_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    writer = SummaryWriter(log_root) if is_main_process() else None

    model = build_student_image_encoder(args.student_arch).to(device)
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 1
    resume_batch_idx = -1
    global_step = 0
    best_val_loss = float("inf")
    resume_path = resolve_resume_path(args)
    if resume_path:
        resume_state = load_checkpoint(resume_path, args, model, optimizer, scheduler, scaler, device)
        start_epoch = resume_state["start_epoch"]
        resume_batch_idx = resume_state["resume_batch_idx"]
        global_step = resume_state["global_step"]
        best_val_loss = resume_state["best_val_loss"]
        if is_main_process():
            print(f"Resumed from {resume_path} at epoch {start_epoch}, after batch {resume_batch_idx}, global_step {global_step}")

    if is_main_process():
        params = sum(p.numel() for p in model.parameters())
        print(f"student_arch={args.student_arch} params={params}")
        print(f"global_batch_size={args.batch_size} per_rank_batch_size={per_rank_batch_size} world_size={get_world_size()}")

    log_window_samples = 0
    log_window_start = time.time()
    last_val_loss = None

    for epoch in range(start_epoch, args.epochs + 1):
        if is_main_process():
            print(f"------start epoch {epoch}------")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        for batch_idx, (imgs, target_feats, _) in enumerate(train_loader):
            if epoch == start_epoch and batch_idx <= resume_batch_idx:
                continue

            global_step += 1
            imgs = imgs.to(device, non_blocking=True)
            target_feats = target_feats.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_feats = model(imgs)
                loss, loss_dict = compute_losses(args, pred_feats, target_feats)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            reduced_losses = reduce_loss_dict(loss_dict)
            log_window_samples += imgs.shape[0] * get_world_size()

            if is_main_process() and global_step % args.print_iters == 0:
                elapsed = max(time.time() - log_window_start, 1e-6)
                samples_per_sec = log_window_samples / elapsed
                memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(
                    f"Train Epoch: {epoch} [{batch_idx * imgs.shape[0] * get_world_size()}/{len(train_loader.dataset)}] "
                    f"total={reduced_losses['total_loss']:.6f} mse={reduced_losses['mse_loss']:.6f} "
                    f"cos={reduced_losses['cosine_loss']:.6f} imgs/s={samples_per_sec:.2f} max_mem_gb={memory_gb:.2f}"
                )
                writer.add_scalar("train/total_loss", reduced_losses["total_loss"], global_step)
                writer.add_scalar("train/mse_loss", reduced_losses["mse_loss"], global_step)
                writer.add_scalar("train/cosine_loss", reduced_losses["cosine_loss"], global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/imgs_per_sec", samples_per_sec, global_step)
                writer.add_scalar("train/max_memory_gb", memory_gb, global_step)
                log_window_samples = 0
                log_window_start = time.time()

            if is_main_process() and args.save_iters > 0 and global_step % args.save_iters == 0:
                iter_path = os.path.join(save_root, f"iter_{global_step}.pt")
                save_checkpoint(iter_path, args, model, optimizer, scheduler, scaler, epoch, batch_idx, global_step, best_val_loss, False)
                save_checkpoint(os.path.join(save_root, "last.pt"), args, model, optimizer, scheduler, scaler, epoch, batch_idx, global_step, best_val_loss, False)

            if args.eval_iters > 0 and global_step % args.eval_iters == 0:
                val_loss = evaluate(args, model, val_loader)
                last_val_loss = val_loss
                model.train()
                if is_main_process():
                    writer.add_scalar("val/total_loss", val_loss, global_step)

            if STOP_REQUESTED:
                if is_main_process():
                    save_checkpoint(os.path.join(save_root, "last.pt"), args, model, optimizer, scheduler, scaler, epoch, batch_idx, global_step, best_val_loss, False)
                    if writer is not None:
                        writer.close()
                if is_dist_avail_and_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
                sys.exit(143)

        if is_dist_avail_and_initialized():
            dist.barrier()

        last_val_loss = evaluate(args, model, val_loader)
        scheduler.step()

        if is_main_process():
            writer.add_scalar("val/total_loss", last_val_loss, global_step)
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                save_checkpoint(os.path.join(save_root, "best_val.pt"), args, model, optimizer, scheduler, scaler, epoch, len(train_loader) - 1, global_step, best_val_loss, True)
            save_checkpoint(os.path.join(save_root, "last.pt"), args, model, optimizer, scheduler, scaler, epoch, len(train_loader) - 1, global_step, best_val_loss, True)
            torch.save(model_state_dict(model), os.path.join(save_root, "iter_final.pth"))

    if is_main_process():
        torch.save(model_state_dict(model), os.path.join(save_root, "iter_final.pth"))
        if writer is not None:
            writer.close()

    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_option())
