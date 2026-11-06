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
from db_sigreg.diagnostics import embedding_metrics, gradient_fidelity
from db_sigreg.gpu_data import build_gpu_loaders
from db_sigreg.models import SSLModel
from db_sigreg.regularizers import DoubleBufferSIGReg, SIGReg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DB-SIGReg / SIGReg training")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "stl10", "fake"], default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--tag", default=None, help="Experiment suite label stored with the results.")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/runs.jsonl"),
        help="Tracked JSONL file; one summary line is appended per finished run.",
    )
    parser.add_argument("--backbone", choices=["cnn", "resnet18", "tiny_vit"], default="cnn")
    parser.add_argument("--loss", choices=["dbsigreg", "sigreg"], default="dbsigreg")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Train for this many optimizer steps instead of --epochs (fixed-steps control).",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--gpu-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep CIFAR on the GPU and augment there (default: on for CIFAR when running on CUDA).",
    )
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--proj-hidden-dim", type=int, default=512)
    parser.add_argument("--axes", type=int, default=128)
    parser.add_argument("--knots", type=int, default=5)
    parser.add_argument(
        "--sigreg-refresh",
        type=int,
        default=1,
        help="SIGReg only: redraw axes every N forward passes; 0 keeps them fixed forever.",
    )
    parser.add_argument("--norm", choices=["bn", "gn"], default="bn", help="gn = GroupNorm/LayerNorm, batch-independent.")
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
    parser.add_argument(
        "--fidelity-every",
        type=int,
        default=0,
        help="DB-SIGReg only: every N optimizer steps, compare gradient estimators against the exact window gradient.",
    )
    parser.add_argument(
        "--fidelity-pool",
        type=int,
        default=4096,
        help="Images in the independent pool used as the population reference (0 disables it).",
    )
    parser.add_argument("--metric-axes", type=int, default=256, help="Fixed axes for the eval-set ECF error.")
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
    tracker: dict,
    fidelity_stream: "BatchStream | None" = None,
) -> int:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Timing is measured over log intervals with one sync each, so the CPU can
    # keep queueing work while the GPU runs (no per-step synchronize).
    _sync(device)
    last_log = time.perf_counter()
    samples_since_log = 0
    steps_since_log = 0

    for step, (views, labels) in enumerate(train_loader, start=1):
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
            tracker["optimizer_steps"] += 1
            if args.total_steps and tracker["optimizer_steps"] >= args.total_steps:
                break
            if (
                fidelity_stream is not None
                and isinstance(regularizer, DoubleBufferSIGReg)
                and regularizer.mature
                and tracker["optimizer_steps"] % args.fidelity_every == 0
            ):
                fid = run_fidelity(model, regularizer, fidelity_stream, args, device)
                for key, value in fid.items():
                    writer.add_scalar(f"fidelity/{key}", value, global_step)
                    tracker["fidelity"].setdefault(key, []).append(value)

        global_step += 1
        samples_since_log += labels.shape[0]
        steps_since_log += 1

        if global_step % args.log_every == 0 or step == 1:
            _sync(device)
            elapsed = time.perf_counter() - last_log
            iter_time = elapsed / steps_since_log
            samples_per_sec = samples_since_log / max(elapsed, 1e-9)
            if global_step > args.log_every:  # skip warm-up (cudnn autotune, compile, worker start)
                tracker["iter_time"] += elapsed
                tracker["iter_samples"] += samples_since_log
            samples_since_log = 0
            steps_since_log = 0
            writer.add_scalar("train/loss", loss.item() * args.accum_steps, global_step)
            writer.add_scalar("train/ssl_loss", ssl_loss.item(), global_step)
            writer.add_scalar("train/invariance_loss", inv_loss.item(), global_step)
            writer.add_scalar("train/sigreg_loss", sig_loss.item(), global_step)
            writer.add_scalar("train/sigreg_metric", float(sig_stats.loss_metric), global_step)
            writer.add_scalar("train/sigreg_ecf_error", float(sig_stats.ecf_error), global_step)
            writer.add_scalar("train/sigreg_optimization_loss", float(sig_stats.optimization_loss), global_step)
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
                tracker["peak_memory_mb"] = max(tracker["peak_memory_mb"], peak_mb)
                writer.add_scalar("perf/peak_memory_mb", peak_mb, global_step)
            last_log = time.perf_counter()

    return global_step


def build_cpu_loaders(args: argparse.Namespace, device: torch.device):
    train_ds, eval_ds, num_classes = build_datasets(
        args.dataset,
        args.data_dir,
        args.image_size,
        args.views,
        args.limit_train,
        args.limit_eval,
        args.download,
    )
    common = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    eval_loader = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, **common)
    return train_loader, eval_loader, num_classes, len(train_ds), len(eval_ds)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class BatchStream:
    """Endless iterator over a DataLoader, used to draw fidelity windows."""

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self._it = iter(loader)

    def next(self) -> torch.Tensor:
        try:
            views, _ = next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            views, _ = next(self._it)
        return views.to(self.device, non_blocking=True)

    def window(self, k: int) -> list[torch.Tensor]:
        return [self.next() for _ in range(k)]


def run_fidelity(
    model: nn.Module,
    regularizer: DoubleBufferSIGReg,
    stream: BatchStream,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    params = [p for p in list(model.encoder.parameters()) + list(model.projector.parameters()) if p.requires_grad]
    window = stream.window(args.accum_steps)
    independent = stream.window(args.accum_steps)
    pool_batches = math.ceil(args.fidelity_pool / args.batch_size) if args.fidelity_pool > 0 else 0
    pool = stream.window(pool_batches) if pool_batches else None
    return gradient_fidelity(
        model,
        params,
        window,
        independent,
        axes=regularizer.active_axes,
        t=regularizer.t.to(device),
        target=regularizer.target.to(device),
        weights=regularizer.weights.to(device),
        active_cos=regularizer.active_cos,
        active_sin=regularizer.active_sin,
        pool=pool,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    metric_axes: torch.Tensor,
    metric_reg: nn.Module,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    projections = []
    for views, labels in loader:
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            emb, proj = model(views)
            logits = model.probe(emb)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
        projections.append(proj.reshape(-1, proj.shape[-1]).float())
    acc = correct / max(total, 1)
    metrics = embedding_metrics(
        torch.cat(projections),
        metric_axes,
        metric_reg.t.to(device),
        metric_reg.target.to(device),
        metric_reg.weights.to(device),
    )
    metrics["probe_acc"] = acc
    for key, value in metrics.items():
        writer.add_scalar(f"eval/{key}", value, epoch)
    return metrics


def main() -> None:
    args = parse_args()
    if args.swap_steps is None:
        args.swap_steps = args.accum_steps
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    amp_enabled, amp_dtype = choose_amp(device, args.amp)
    run_dir = make_run_dir(args)

    if args.gpu_data is None:
        args.gpu_data = device.type == "cuda" and args.dataset in ("cifar10", "cifar100")
    if args.gpu_data and args.dataset not in ("cifar10", "cifar100"):
        raise SystemExit("--gpu-data supports cifar10 and cifar100 only")

    if args.gpu_data:
        loader_kwargs = dict(
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            views=args.views,
            image_size=args.image_size,
            limit_train=args.limit_train,
            limit_eval=args.limit_eval,
            download=args.download,
            device=device,
        )
        train_loader, eval_loader, num_classes, n_train, n_eval = build_gpu_loaders(
            args.dataset, args.data_dir, seed=args.seed, **loader_kwargs
        )
    else:
        train_loader, eval_loader, num_classes, n_train, n_eval = build_cpu_loaders(args, device)

    model = SSLModel(
        backbone=args.backbone,
        image_size=args.image_size,
        emb_dim=args.emb_dim,
        proj_dim=args.proj_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        num_classes=num_classes,
        norm=args.norm,
    ).to(device)
    if args.compile:
        model = torch.compile(model)

    if args.loss == "sigreg":
        regularizer: nn.Module = SIGReg(axes=args.axes, knots=args.knots, refresh_every=args.sigreg_refresh).to(device)
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
    print(f"Train samples: {n_train} | Eval samples: {n_eval} | Data on GPU: {args.gpu_data}")
    print(f"Effective optimizer batch: {args.batch_size * args.accum_steps}")
    if args.loss == "dbsigreg" and args.db_stat_mode == "detached" and args.swap_steps != args.accum_steps:
        print("Note: DB-SIGReg is easiest to interpret when --swap-steps equals --accum-steps.")

    fidelity_stream = None
    if args.fidelity_every > 0 and args.loss == "dbsigreg":
        if args.gpu_data:
            fidelity_loader, *_ = build_gpu_loaders(args.dataset, args.data_dir, seed=args.seed + 1, **loader_kwargs)
        else:
            fidelity_loader = DataLoader(
                train_loader.dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=min(args.num_workers, 2),
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
        fidelity_stream = BatchStream(fidelity_loader, device)

    # Fixed, seeded axes so the eval-set ECF error is comparable across runs.
    metric_reg = SIGReg(axes=args.metric_axes, knots=args.knots)
    gen = torch.Generator().manual_seed(1234)
    metric_axes = torch.randn(args.proj_dim, args.metric_axes, generator=gen)
    metric_axes = (metric_axes / metric_axes.norm(dim=0, keepdim=True)).to(device)

    tracker: dict = {
        "optimizer_steps": 0,
        "iter_time": 0.0,
        "iter_samples": 0,
        "peak_memory_mb": 0.0,
        "fidelity": {},
    }
    history: list[dict[str, float]] = []
    start_time = time.time()
    if args.total_steps:
        steps_per_epoch = max(len(train_loader) // args.accum_steps, 1)
        args.epochs = math.ceil(args.total_steps / steps_per_epoch)
        print(f"Fixed {args.total_steps} optimizer steps -> {args.epochs} epochs")
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
                tracker=tracker,
                fidelity_stream=fidelity_stream,
            )
            done = bool(args.total_steps) and tracker["optimizer_steps"] >= args.total_steps
            if epoch % args.eval_every == 0 or epoch == args.epochs or done:
                metrics = evaluate(
                    model, eval_loader, writer, device, amp_enabled, amp_dtype, epoch, metric_axes, metric_reg
                )
                acc = metrics["probe_acc"]
                best_acc = max(best_acc, acc)
                history.append({"epoch": epoch, **metrics})
                print(
                    f"epoch={epoch:03d} probe_acc={acc:.4f} best={best_acc:.4f} "
                    f"ecf_error={metrics['ecf_error']:.5f} axis_std={metrics['axis_std']:.4f}"
                )
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
            if done:
                break
    finally:
        writer.close()

    write_results(args, run_dir, history, tracker, time.time() - start_time)


def write_results(
    args: argparse.Namespace,
    run_dir: Path,
    history: list[dict[str, float]],
    tracker: dict,
    wall_time: float,
) -> None:
    final = history[-1] if history else {}
    fidelity = {}
    for key, values in tracker["fidelity"].items():
        if values:
            ordered = sorted(values)
            fidelity[f"fidelity_{key}_mean"] = sum(values) / len(values)
            fidelity[f"fidelity_{key}_median"] = ordered[len(ordered) // 2]
    summary = {
        "run": run_dir.name,
        "tag": args.tag,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "final_probe_acc": final.get("probe_acc"),
        "best_probe_acc": max((h["probe_acc"] for h in history), default=None),
        "final_ecf_error": final.get("ecf_error"),
        "final_axis_std": final.get("axis_std"),
        "final_dim_std": final.get("dim_std"),
        "samples_per_sec": tracker["iter_samples"] / tracker["iter_time"] if tracker["iter_time"] else None,
        "peak_memory_mb": tracker["peak_memory_mb"] or None,
        "optimizer_steps": tracker["optimizer_steps"],
        "wall_time_sec": wall_time,
        **fidelity,
        "history": history,
        "fidelity_trace": tracker["fidelity"],
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with args.results_file.open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(f"Appended summary to {args.results_file}")


if __name__ == "__main__":
    main()
