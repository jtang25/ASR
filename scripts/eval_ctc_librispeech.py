#!/usr/bin/env python3
"""Evaluate a CTC checkpoint on LibriSpeech splits with optional KenLM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Allow running as: python3 scripts/eval_ctc_librispeech.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoding import build_lm_decoder
from model import ConformerASR
from preprocessing import get_dataloader
from tokenizer import BLANK_IDX, CharTokenizer, SentencePieceTokenizer
from train import _autocast_dtype, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CTC ASR checkpoint on LibriSpeech.")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--splits", nargs="+", default=["test-clean", "test-other"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    p.add_argument("--persistent-workers", action="store_true", default=True)
    p.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers")
    p.add_argument("--download", action="store_true", default=False)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--beam-size", type=int, default=20)
    p.add_argument("--beam-token-prune", type=int, default=0)
    p.add_argument("--lm-path", default=None, help="Optional KenLM .arpa/.bin path.")
    p.add_argument("--lm-alpha", type=float, default=0.5)
    p.add_argument("--lm-beta", type=float, default=1.0)
    p.add_argument("--lm-beam-width", type=int, default=128)
    return p.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(cfg: dict):
    tok_type = cfg.get("tokenizer", "sp")
    if tok_type == "sp":
        sp_model = cfg.get("sp_model")
        if not sp_model:
            raise ValueError("Checkpoint config has tokenizer='sp' but no sp_model path.")
        return SentencePieceTokenizer(sp_model)
    return CharTokenizer()


def _ctc_decoder_vocab(tokenizer) -> list[str]:
    return [str(tokenizer.id_to_piece(i)) for i in range(int(tokenizer.vocab_size))]


def _build_model(cfg: dict, vocab_size: int, device: torch.device) -> ConformerASR:
    return ConformerASR(
        n_mels=int(cfg.get("n_mels", 80)),
        d_model=int(cfg.get("d_model", 256)),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 12)),
        vocab_size=int(vocab_size),
        conv_kernel_size=int(cfg.get("conv_kernel", 32)),
        max_len=int(cfg.get("max_len", 2048)),
        ffn_dropout=float(cfg.get("dropout", 0.1)),
        attn_dropout=float(cfg.get("dropout", 0.1)),
        conv_dropout=float(cfg.get("dropout", 0.1)),
        streaming_chunk_size=int(cfg.get("streaming_chunk_size", 0)),
        streaming_left_context_chunks=int(cfg.get("streaming_left_context_chunks", -1)),
        streaming_right_context=int(cfg.get("streaming_right_context", 0)),
        streaming_causal_conv=bool(cfg.get("streaming_causal_conv", False)),
    ).to(device)


def main() -> None:
    args = parse_args()
    if args.beam_size < 1:
        raise ValueError("--beam-size must be >= 1")
    if args.beam_token_prune < 0:
        raise ValueError("--beam-token-prune must be >= 0")
    if args.lm_beam_width < 1:
        raise ValueError("--lm-beam-width must be >= 1")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = _resolve_device(args.device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or ckpt.get("args", {}) or {}
    if cfg.get("loss_type", "ctc") != "ctc":
        raise ValueError("This evaluator only supports CTC checkpoints.")

    tokenizer = _load_tokenizer(cfg)
    cfg_vocab_size = int(cfg.get("vocab_size", tokenizer.vocab_size))
    if cfg_vocab_size != int(tokenizer.vocab_size):
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match checkpoint vocab_size ({cfg_vocab_size})."
        )

    model = _build_model(cfg=cfg, vocab_size=cfg_vocab_size, device=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(
            f"[WARN] Checkpoint/model mismatch "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
    model.eval()

    precision = cfg.get("precision", "bf16") if args.precision == "auto" else args.precision
    autocast_dtype = _autocast_dtype(str(precision))

    lm_decoder = None
    if args.lm_path:
        lm_decoder = build_lm_decoder(
            vocab=_ctc_decoder_vocab(tokenizer),
            lm_path=args.lm_path,
            blank_idx=BLANK_IDX,
            alpha=float(args.lm_alpha),
            beta=float(args.lm_beta),
        )

    print(f"checkpoint: {ckpt_path}")
    print(f"device: {device}")
    print(f"tokenizer: {cfg.get('tokenizer', 'char')} (vocab={tokenizer.vocab_size})")
    print(f"decode: {'ctc+lm' if lm_decoder is not None else 'ctc-beam'}")

    token_prune = args.beam_token_prune if args.beam_token_prune > 0 else None

    for split in args.splits:
        loader = get_dataloader(
            root=args.data_root,
            split=split,
            batch_size=args.batch_size,
            n_mels=int(cfg.get("n_mels", 80)),
            augment=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            download=args.download,
            distributed=False,
            tokenizer=tokenizer,
        )

        val_loss, wer, _wer_greedy, _wer_beam = evaluate(
            model=model,
            loader=loader,
            device=device,
            loss_type="ctc",
            tokenizer=tokenizer,
            beam_size=args.beam_size,
            token_prune=token_prune,
            lm_decoder=lm_decoder,
            lm_beam_width=args.lm_beam_width,
            autocast_dtype=autocast_dtype,
            is_distributed=False,
        )
        print(f"{split}: WER={wer * 100.0:.2f}% loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
