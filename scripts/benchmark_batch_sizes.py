#!/usr/bin/env python3
"""Short fixed-step benchmark for RNNT training batch sizes.

Runs the same model/data/settings for multiple per-GPU batch sizes and reports:
  - steps/sec
  - torch.cuda.max_memory_reserved peak
  - NaN/Inf loss stability
"""

from __future__ import annotations

import argparse
import gc
import itertools
import math
import time
from types import SimpleNamespace
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Allow running as: python3 scripts/benchmark_batch_sizes.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ConformerASR, ConformerTransducer
from preprocessing import get_dataloader
from tokenizer import CharTokenizer, SentencePieceTokenizer
from train import (
    PaperTransformerSchedule,
    WarmupCosineScheduler,
    _autocast_dtype,
    _cleanup,
    _configure_runtime,
    _init_distributed,
    _set_seed,
    combined_loader,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark RNNT batch sizes on fixed train steps.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint to load model/init state from.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--data-root", default="./data")
    p.add_argument(
        "--train-splits",
        nargs="+",
        default=["train-clean-100", "train-clean-360", "train-other-500"],
    )
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 96, 128])
    p.add_argument("--steps", type=int, default=120, help="Fixed optimizer steps per benchmark run.")
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--prefetch-factor", type=int, default=6)
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    p.add_argument("--persistent-workers", action="store_true", default=True)
    p.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers")
    p.add_argument("--seed", type=int, default=1337)

    # Streaming config to benchmark (same across batch sizes).
    # Default behavior is to use checkpoint streaming settings.
    p.add_argument(
        "--streaming-mode",
        action="store_true",
        dest="streaming_mode",
        default=None,
        help="Force streaming mode on for benchmark model.",
    )
    p.add_argument(
        "--no-streaming-mode",
        action="store_false",
        dest="streaming_mode",
        help="Force offline mode (no streaming mask) for benchmark model.",
    )
    p.add_argument("--streaming-chunk-size", type=int, default=16)
    p.add_argument("--streaming-left-context-chunks", type=int, default=32)
    p.add_argument("--streaming-right-context", type=int, default=4)
    p.add_argument("--streaming-causal-conv", action="store_true", default=False)
    p.add_argument("--no-streaming-causal-conv", action="store_false", dest="streaming_causal_conv")

    # Training loss/optimizer behavior.
    p.add_argument("--variational-noise", type=float, default=0.0)
    p.add_argument("--rnnt-loss-device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--rnnt-max-batch-tu", type=int, default=2_000_000)
    p.add_argument("--rnnt-fused-log-softmax", action="store_true", default=False)
    p.add_argument("--reset-scheduler-on-resume", action="store_true", default=False)
    return p.parse_args()


def _resolve_streaming_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    if args.streaming_mode is None:
        ckpt_streaming = bool(cfg.get("streaming_mode", False) or int(cfg.get("streaming_chunk_size", 0)) > 0)
        if ckpt_streaming:
            chunk = int(cfg.get("streaming_chunk_size", 8))
            if chunk <= 0:
                chunk = 8
            return {
                "enabled": True,
                "chunk_size": chunk,
                "left_chunks": int(cfg.get("streaming_left_context_chunks", -1)),
                "right_context": int(cfg.get("streaming_right_context", 0)),
                "causal_conv": bool(cfg.get("streaming_causal_conv", False)),
            }
        return {
            "enabled": False,
            "chunk_size": 0,
            "left_chunks": -1,
            "right_context": 0,
            "causal_conv": False,
        }

    if args.streaming_mode:
        return {
            "enabled": True,
            "chunk_size": int(args.streaming_chunk_size),
            "left_chunks": int(args.streaming_left_context_chunks),
            "right_context": int(args.streaming_right_context),
            "causal_conv": bool(args.streaming_causal_conv),
        }
    return {
        "enabled": False,
        "chunk_size": 0,
        "left_chunks": -1,
        "right_context": 0,
        "causal_conv": False,
    }


def _build_model(cfg: dict, device: torch.device, stream_cfg: dict):
    loss_type = cfg.get("loss_type", "rnnt")
    if loss_type == "ctc":
        model = ConformerASR(
            n_mels=int(cfg.get("n_mels", 80)),
            d_model=int(cfg.get("d_model", 256)),
            num_heads=int(cfg.get("num_heads", 4)),
            num_layers=int(cfg.get("num_layers", 12)),
            vocab_size=int(cfg.get("vocab_size", 29)),
            conv_kernel_size=int(cfg.get("conv_kernel", 31)),
            max_len=int(cfg.get("max_len", 2048)),
            ffn_dropout=float(cfg.get("dropout", 0.1)),
            attn_dropout=float(cfg.get("dropout", 0.1)),
            conv_dropout=float(cfg.get("dropout", 0.1)),
            streaming_chunk_size=int(stream_cfg["chunk_size"] if stream_cfg["enabled"] else 0),
            streaming_left_context_chunks=int(stream_cfg["left_chunks"] if stream_cfg["enabled"] else -1),
            streaming_right_context=int(stream_cfg["right_context"] if stream_cfg["enabled"] else 0),
            streaming_causal_conv=bool(stream_cfg["causal_conv"] if stream_cfg["enabled"] else False),
        ).to(device)
    else:
        model = ConformerTransducer(
            n_mels=int(cfg.get("n_mels", 80)),
            encoder_dim=int(cfg.get("d_model", 256)),
            num_heads=int(cfg.get("num_heads", 4)),
            num_encoder_layers=int(cfg.get("num_layers", 12)),
            vocab_size=int(cfg.get("vocab_size", 1024)),
            conv_kernel_size=int(cfg.get("conv_kernel", 31)),
            max_len=int(cfg.get("max_len", 2048)),
            ffn_dropout=float(cfg.get("dropout", 0.1)),
            attn_dropout=float(cfg.get("dropout", 0.1)),
            conv_dropout=float(cfg.get("dropout", 0.1)),
            pred_embed_dim=int(cfg.get("pred_embed_dim", 256)),
            pred_hidden_dim=int(cfg.get("pred_hidden_dim", 640)),
            pred_num_layers=int(cfg.get("pred_num_layers", 1)),
            joint_dim=int(cfg.get("joint_dim", 640)),
            blank_idx=0,
            streaming_chunk_size=int(stream_cfg["chunk_size"] if stream_cfg["enabled"] else 0),
            streaming_left_context_chunks=int(stream_cfg["left_chunks"] if stream_cfg["enabled"] else -1),
            streaming_right_context=int(stream_cfg["right_context"] if stream_cfg["enabled"] else 0),
            streaming_causal_conv=bool(stream_cfg["causal_conv"] if stream_cfg["enabled"] else False),
        ).to(device)
    return model, loss_type


def _build_tokenizer(cfg: dict):
    tok_type = cfg.get("tokenizer", "sp")
    if tok_type == "sp":
        return SentencePieceTokenizer(cfg.get("sp_model", "./tokenizer/sp_1k.model"))
    return CharTokenizer()


def _build_optimizer(model, cfg: dict, device: torch.device):
    optimizer_name = cfg.get("optimizer", "adam")
    opt_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    base = {
        "lr": float(cfg.get("lr", 3e-4)),
        "weight_decay": float(cfg.get("weight_decay", 1e-6)),
        "betas": (0.9, 0.98),
        "eps": 1e-9,
    }
    candidates = [dict(base)]
    if device.type == "cuda":
        fused = bool(cfg.get("fused_optimizer", True))
        if fused:
            candidates = [{**base, "fused": True}, {**base, "foreach": True}, dict(base)]
        else:
            candidates = [{**base, "foreach": True}, dict(base)]

    last_err: Exception | None = None
    for kw in candidates:
        try:
            return opt_cls(model.parameters(), **kw)
        except (TypeError, RuntimeError, ValueError) as err:
            last_err = err
    raise RuntimeError(f"Failed to create optimizer. Last error: {last_err}")


def _build_scheduler(optimizer, cfg: dict):
    lr_schedule = cfg.get("lr_schedule", "paper")
    if lr_schedule == "paper":
        return PaperTransformerSchedule(
            optimizer=optimizer,
            d_model=int(cfg.get("d_model", 256)),
            warmup_steps=int(cfg.get("warmup_steps", 10_000)),
            peak_factor=float(cfg.get("paper_peak_factor", 0.05)),
        )
    return WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=int(cfg.get("warmup_steps", 10_000)),
        total_steps=int(cfg.get("total_steps", 100_000)),
        peak_lr=float(cfg.get("lr", 3e-4)),
        min_lr=float(cfg.get("min_lr", 1e-6)),
    )


def _nan_detected(step_rows: list[dict], epoch_loss: float) -> bool:
    if not math.isfinite(epoch_loss):
        return True
    for row in step_rows:
        val = row.get("train_step_loss", float("nan"))
        if isinstance(val, str):
            continue
        if not math.isfinite(float(val)):
            return True
    return False


def _dist_max(value: float, device: torch.device, enabled: bool) -> float:
    if not enabled:
        return value
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _dist_any(flag: bool, device: torch.device, enabled: bool) -> bool:
    if not enabled:
        return bool(flag)
    t = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(int(t.item()))


def _dist_min_int(value: int, device: torch.device, enabled: bool) -> int:
    if not enabled:
        return value
    t = torch.tensor(int(value), device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def main() -> None:
    args = parse_args()

    is_dist, rank, local_rank, world_size, device = _init_distributed(args.device, timeout_sec=300)
    is_main = rank == 0

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {}) or ckpt.get("args", {})

        runtime_ns = SimpleNamespace(
            precision=cfg.get("precision", "bf16"),
            tf32=bool(cfg.get("tf32", True)),
        )
        _configure_runtime(runtime_ns, device, is_main)

        tokenizer = _build_tokenizer(cfg)
        autocast_dtype = _autocast_dtype(cfg.get("precision", "bf16"))
        scaler = GradScaler(device=device.type, enabled=(device.type == "cuda" and cfg.get("precision") == "fp16"))

        if is_main:
            print(f"Benchmark checkpoint: {args.checkpoint}")
            print(f"World size: {world_size} | Device: {device}")
            print(f"Steps per run: {args.steps} | Batch sizes: {args.batch_sizes}")
        stream_cfg = _resolve_streaming_cfg(cfg, args)
        if is_main:
            if stream_cfg["enabled"]:
                print(
                    "Benchmark streaming: enabled "
                    f"(chunk={stream_cfg['chunk_size']} left={stream_cfg['left_chunks']} "
                    f"right={stream_cfg['right_context']} causal_conv={stream_cfg['causal_conv']})"
                )
            else:
                print("Benchmark streaming: disabled (offline)")

        results: list[dict] = []

        for idx, batch_size in enumerate(args.batch_sizes):
            _set_seed(args.seed + rank)

            pairs = [
                get_dataloader(
                    root=args.data_root,
                    split=split,
                    batch_size=batch_size,
                    n_mels=int(cfg.get("n_mels", 80)),
                    augment=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    pin_memory=args.pin_memory,
                    persistent_workers=args.persistent_workers,
                    download=False,
                    distributed=is_dist,
                    rank=rank,
                    world_size=world_size,
                    return_sampler=True,
                    tokenizer=tokenizer,
                )
                for split in args.train_splits
            ]
            train_loaders = [x[0] for x in pairs]
            train_samplers = [x[1] for x in pairs if isinstance(x[1], DistributedSampler)]
            for sampler in train_samplers:
                sampler.set_epoch(0)

            model_raw, loss_type = _build_model(cfg, device, stream_cfg)
            model_raw.load_state_dict(ckpt["model"], strict=True)
            model = model_raw
            if is_dist:
                ddp_kw = {
                    "bucket_cap_mb": int(cfg.get("ddp_bucket_cap_mb", 100)),
                    "gradient_as_bucket_view": bool(cfg.get("ddp_grad_as_bucket_view", True)),
                    "broadcast_buffers": bool(cfg.get("ddp_broadcast_buffers", False)),
                }
                if device.type == "cuda":
                    ddp_kw.update({"device_ids": [local_rank], "output_device": local_rank})
                model = DDP(model_raw, **ddp_kw)

            optimizer = _build_optimizer(model, cfg, device)
            if isinstance(ckpt.get("optimizer"), dict):
                optimizer.load_state_dict(ckpt["optimizer"])

            scheduler = _build_scheduler(optimizer, cfg)
            if args.reset_scheduler_on_resume:
                scheduler.step_count = 0
            elif isinstance(ckpt.get("scheduler"), dict):
                scheduler.load_state_dict(ckpt["scheduler"])

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            if is_dist:
                dist.barrier()

            limited_loader = itertools.islice(combined_loader(train_loaders, shuffle=True), args.steps)
            t0 = time.perf_counter()
            train_loss, step_rows, _num_batches = train_one_epoch(
                model=model,
                loader=limited_loader,
                loader_len=args.steps,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                grad_clip=float(cfg.get("grad_clip", 5.0)),
                accum_steps=args.accum_steps,
                epoch_num=1,
                global_step_offset=0,
                loss_type=loss_type,
                var_noise_std=float(args.variational_noise),
                is_distributed=is_dist,
                is_main=is_main,
                autocast_dtype=autocast_dtype,
                rnnt_fused_log_softmax=bool(args.rnnt_fused_log_softmax),
                rnnt_loss_device=args.rnnt_loss_device,
                rnnt_max_batch_tu=int(args.rnnt_max_batch_tu),
            )
            elapsed = time.perf_counter() - t0

            steps_done = len(step_rows)
            nan_flag = _nan_detected(step_rows, train_loss)
            peak_reserved = float(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0.0

            elapsed_max = _dist_max(elapsed, device, is_dist)
            peak_max = _dist_max(peak_reserved, device, is_dist)
            steps_min = _dist_min_int(steps_done, device, is_dist)
            nan_any = _dist_any(nan_flag, device, is_dist)

            steps_per_sec = float(steps_min) / max(elapsed_max, 1e-9)
            result = {
                "batch_size": int(batch_size),
                "steps": int(steps_min),
                "elapsed_sec": float(elapsed_max),
                "steps_per_sec": float(steps_per_sec),
                "peak_reserved_gib": float(peak_max / (1024 ** 3)),
                "nan_or_inf": bool(nan_any),
                "train_loss": float(train_loss),
            }
            results.append(result)

            if is_main:
                print(
                    f"[batch={batch_size}] steps={result['steps']} "
                    f"time={result['elapsed_sec']:.2f}s "
                    f"steps/s={result['steps_per_sec']:.3f} "
                    f"peak_reserved={result['peak_reserved_gib']:.2f} GiB "
                    f"nan_or_inf={result['nan_or_inf']} "
                    f"train_loss={result['train_loss']:.4f}"
                )

            del model, model_raw, optimizer, scheduler, train_loaders, train_samplers, pairs, step_rows
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if is_dist:
                dist.barrier()

        if is_main:
            print("\nSummary")
            print("batch_size,steps,elapsed_sec,steps_per_sec,peak_reserved_gib,nan_or_inf,train_loss")
            for r in results:
                print(
                    f"{r['batch_size']},{r['steps']},{r['elapsed_sec']:.4f},"
                    f"{r['steps_per_sec']:.4f},{r['peak_reserved_gib']:.4f},"
                    f"{int(r['nan_or_inf'])},{r['train_loss']:.6f}"
                )
    finally:
        _cleanup(is_dist)


if __name__ == "__main__":
    main()
