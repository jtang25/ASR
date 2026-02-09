#!/usr/bin/env python3
"""Decode one sample utterance from a training checkpoint.

Defaults:
  checkpoint = ./checkpoints_conformer_l_4xh200/last.pt
  sample     = first .flac found under ./data/LibriSpeech/dev-other
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Allow running as: python3 scripts/decode_checkpoint_sample.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ConformerASR, ConformerTransducer
from preprocessing import LogMelSpectrogram
from tokenizer import BLANK_IDX, CharTokenizer, SentencePieceTokenizer
from train import decode_eval_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode a sample with a saved ASR checkpoint.")
    p.add_argument(
        "--checkpoint",
        default="./checkpoints_conformer_l_4xh200/last.pt",
        help="Path to checkpoint .pt file.",
    )
    p.add_argument(
        "--sample-path",
        default=None,
        help="Optional .flac path. If omitted, uses first file from data-root/LibriSpeech/<split>.",
    )
    p.add_argument(
        "--data-root",
        default="./data",
        help="Dataset root used when --sample-path is omitted.",
    )
    p.add_argument(
        "--split",
        default="dev-other",
        help="LibriSpeech split used when --sample-path is omitted.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for decode.",
    )
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_first_flac(data_root: str, split: str) -> str | None:
    split_root = Path(data_root) / "LibriSpeech" / split
    if not split_root.exists():
        return None
    flacs = sorted(split_root.rglob("*.flac"))
    return str(flacs[0]) if flacs else None


def load_tokenizer(cfg: dict):
    tok_type = cfg.get("tokenizer", "sp")
    if tok_type == "sp":
        sp_model = cfg.get("sp_model")
        if not sp_model:
            raise ValueError("Checkpoint config has tokenizer='sp' but no sp_model path.")
        return SentencePieceTokenizer(sp_model)
    return CharTokenizer()


def load_model(cfg: dict, device: torch.device):
    loss_type = cfg.get("loss_type", "rnnt")
    streaming_chunk_size = int(cfg.get("streaming_chunk_size", 0))
    streaming_left_context_chunks = int(cfg.get("streaming_left_context_chunks", -1))
    streaming_right_context = int(cfg.get("streaming_right_context", 0))
    streaming_causal_conv = bool(cfg.get("streaming_causal_conv", False))
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
            streaming_chunk_size=streaming_chunk_size,
            streaming_left_context_chunks=streaming_left_context_chunks,
            streaming_right_context=streaming_right_context,
            streaming_causal_conv=streaming_causal_conv,
        )
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
            blank_idx=BLANK_IDX,
            streaming_chunk_size=streaming_chunk_size,
            streaming_left_context_chunks=streaming_left_context_chunks,
            streaming_right_context=streaming_right_context,
            streaming_causal_conv=streaming_causal_conv,
        )
    return model.to(device), loss_type


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or ckpt.get("args", {})

    tokenizer = load_tokenizer(cfg)
    model, loss_type = load_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sample_path = args.sample_path or find_first_flac(args.data_root, args.split)
    if sample_path is None:
        raise RuntimeError(
            f"No .flac found under {Path(args.data_root) / 'LibriSpeech' / args.split}"
        )

    mel_extractor = LogMelSpectrogram(n_mels=int(cfg.get("n_mels", 80)))
    ref, hyp = decode_eval_sample(
        model=model,
        device=device,
        loss_type=loss_type,
        tokenizer=tokenizer,
        sample_path=sample_path,
        mel_extractor=mel_extractor,
        beam_size=10,
        token_prune=None,
    )

    print(f"checkpoint: {ckpt_path}")
    print(f"device    : {device}")
    print(f"sample    : {sample_path}")
    if ref:
        print(f"ref       : {ref}")
    print(f"hyp       : {hyp}")


if __name__ == "__main__":
    main()
