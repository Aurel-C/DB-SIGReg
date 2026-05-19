from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from db_sigreg.data import build_datasets
from db_sigreg.models import SSLModel
from db_sigreg.regularizers import DoubleBufferSIGReg, SIGReg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DB-SIGReg / SIGReg training")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "stl10", "fake"], default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--backbone", choices=["cnn", "resnet18", "tiny_vit"], default="cnn")
    parser.add_argument("--loss", choices=["dbsigreg", "sigreg"], default="dbsigreg")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--proj-hidden-dim", type=int, default=512)
    parser.add_argument("--axes", type=int, default=128)
    parser.add_argument("--knots", type=int, default=17)
    parser.add_argument("--lambda-sigreg", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-7)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--swap-steps", type=int, default=None)
    parser.add_argument("--db-stat-mode", choices=["detached", "include_current"], default="detached")
    parser.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def choose_amp(device: torch.device, mode: str) -> tuple[bool, torch.dtype]:
    if device.type != "cuda" or mode == "off":
        return False, torch.float32
    if mode == "bf16":
        return True, torch.bfloat16
    if mode == "fp16":
        return True, torch.float16
    major, _ = torch.cuda.get_device_capability(device)
    return True, torch.bfloat16 if major >= 8 else torch.float16


def make_run_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = args.name or f"{args.dataset}-{args.backbone}-{args.loss}-bs{args.batch_size}-acc{args.accum_steps}"
    run_dir = args.run_dir / f"{stamp}-{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_one_epoch(
    *,
    model: nn.Module,
    regularizer: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    writer: SummaryWriter,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    global_step: int,
) -> int:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_log = time.perf_counter()

    for step, (views, labels) in enumerate(train_loader, start=1):
        iter_start = time.perf_counter()
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            emb, proj = model(views)
            inv_loss = (proj - proj.mean(dim=1, keepdim=True)).square().mean()
            sig_loss, sig_stats = regularizer(proj)
            ssl_loss = (1.0 - args.lambda_sigreg) * inv_loss + args.lambda_sigreg * sig_loss
            logits = model.probe(emb.detach())
            repeated_labels = labels.repeat_interleave(args.views)
            probe_loss = F.cross_entropy(logits, repeated_labels)
            loss = (ssl_loss + probe_loss) / args.accum_steps

        scaler.scale(loss).backward()
        if isinstance(regularizer, DoubleBufferSIGReg):
            regularizer.maybe_swap_buffers()

        should_step = step % args.accum_steps == 0 or step == len(train_loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        iter_time = time.perf_counter() - iter_start
        samples_per_sec = labels.shape[0] / max(iter_time, 1e-9)
        global_step += 1

        if global_step % args.log_every == 0 or step == 1:
            elapsed = time.perf_counter() - last_log
            writer.add_scalar("train/loss", loss.item() * args.accum_steps, global_step)
            writer.add_scalar("train/ssl_loss", ssl_loss.item(), global_step)
            writer.add_scalar("train/invariance_loss", inv_loss.item(), global_step)
            writer.add_scalar("train/sigreg_loss", sig_loss.item(), global_step)
            writer.add_scalar("train/sigreg_metric", sig_stats.loss_metric, global_step)
            writer.add_scalar("train/sigreg_ecf_error", sig_stats.ecf_error, global_step)
            writer.add_scalar("train/sigreg_optimization_loss", sig_stats.optimization_loss, global_step)
            writer.add_scalar("train/probe_loss", probe_loss.item(), global_step)
            writer.add_scalar("buffer/loss_count", sig_stats.loss_count, global_step)
            writer.add_scalar("buffer/active_count", sig_stats.active_count, global_step)
            writer.add_scalar("buffer/shadow_count", sig_stats.shadow_count, global_step)
            writer.add_scalar("buffer/swaps", sig_stats.swaps, global_step)
            writer.add_scalar("buffer/mature", float(sig_stats.mature), global_step)
            writer.add_scalar("perf/samples_per_sec", samples_per_sec, global_step)
            writer.add_scalar("perf/iter_time_sec", iter_time, global_step)
            writer.add_scalar("perf/log_interval_sec", elapsed, global_step)
            if device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                writer.add_scalar("perf/peak_memory_mb", peak_mb, global_step)
            last_log = time.perf_counter()

    return global_step


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    epoch: int,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for views, labels in loader:
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            emb, _ = model(views)
            logits = model.probe(emb)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    acc = correct / max(total, 1)
    writer.add_scalar("eval/probe_acc", acc, epoch)
    return acc


def main() -> None:
    args = parse_args()
    if args.swap_steps is None:
        args.swap_steps = args.accum_steps
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    amp_enabled, amp_dtype = choose_amp(device, args.amp)
    run_dir = make_run_dir(args)

    train_ds, eval_ds, num_classes = build_datasets(
        args.dataset,
        args.data_dir,
        args.image_size,
        args.views,
        args.limit_train,
        args.limit_eval,
        args.download,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = SSLModel(
        backbone=args.backbone,
        image_size=args.image_size,
        emb_dim=args.emb_dim,
        proj_dim=args.proj_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        num_classes=num_classes,
    ).to(device)
    if args.compile:
        model = torch.compile(model)

    if args.loss == "sigreg":
        regularizer: nn.Module = SIGReg(axes=args.axes, knots=args.knots).to(device)
    else:
        regularizer = DoubleBufferSIGReg(
            axes=args.axes,
            knots=args.knots,
            swap_steps=args.swap_steps,
            stat_mode=args.db_stat_mode,
        ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": model.projector.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": model.probe.parameters(), "lr": args.probe_lr, "weight_decay": args.probe_weight_decay},
        ]
    )
    scaler = GradScaler(device=device.type, enabled=amp_enabled and amp_dtype == torch.float16)
    writer = SummaryWriter(run_dir)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    print(f"Run directory: {run_dir}")
    print(f"Device: {device} | AMP: {amp_dtype if amp_enabled else 'off'}")
    print(f"Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}")
    print(f"Effective optimizer batch: {args.batch_size * args.accum_steps}")
    if args.loss == "dbsigreg" and args.db_stat_mode == "detached" and args.swap_steps != args.accum_steps:
        print("Note: DB-SIGReg is easiest to interpret when --swap-steps equals --accum-steps.")

    global_step = 0
    best_acc = -math.inf
    try:
        for epoch in range(1, args.epochs + 1):
            global_step = train_one_epoch(
                model=model,
                regularizer=regularizer,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                writer=writer,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                epoch=epoch,
                global_step=global_step,
            )
            if epoch % args.eval_every == 0:
                acc = evaluate(model, eval_loader, writer, device, amp_enabled, amp_dtype, epoch)
                best_acc = max(best_acc, acc)
                print(f"epoch={epoch:03d} eval/probe_acc={acc:.4f} best={best_acc:.4f}")
            checkpoint = {
                "model": model.state_dict(),
                "regularizer": regularizer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "global_step": global_step,
                "best_acc": best_acc,
            }
            torch.save(checkpoint, run_dir / "last.pt")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
