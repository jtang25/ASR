import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    from pyctcdecode import build_ctcdecoder
except ImportError:  # pragma: no cover - optional dependency
    build_ctcdecoder = None


NEG_INF = float("-inf")


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    beam_size: int = 10,
    blank_idx: int = 0,
    token_prune: Optional[int] = None,
) -> List[int]:
    """
    CTC prefix beam search on a single sequence.

    Args:
        log_probs: (T, V) log probabilities
        beam_size: number of active beams to keep
        blank_idx: CTC blank token index
        token_prune: optional per-frame top-k token pruning
    Returns:
        Decoded token sequence
    """
    if log_probs.numel() == 0:
        return []

    log_probs = log_probs.detach().cpu()
    _, vocab_size = log_probs.shape
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(log_probs.size(0)):
        frame = log_probs[t]
        if token_prune is not None and 0 < token_prune < vocab_size:
            top_vals, top_idx = torch.topk(frame, k=token_prune)
            token_items = list(zip(top_idx.tolist(), top_vals.tolist()))
        else:
            token_items = list(enumerate(frame.tolist()))

        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        for prefix, (p_blank, p_non_blank) in beams.items():
            prefix_total = _log_add(p_blank, p_non_blank)
            last_token = prefix[-1] if prefix else None

            for token, token_logp in token_items:
                if token == blank_idx:
                    n_blank, n_non_blank = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    n_blank = _log_add(n_blank, prefix_total + token_logp)
                    next_beams[prefix] = (n_blank, n_non_blank)
                    continue

                if token == last_token:
                    # Repeating token can either stay on same prefix (from non-blank)
                    # or extend from blank.
                    n_blank, n_non_blank = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    n_non_blank = _log_add(n_non_blank, p_non_blank + token_logp)
                    next_beams[prefix] = (n_blank, n_non_blank)

                    ext_prefix = prefix + (token,)
                    e_blank, e_non_blank = next_beams.get(ext_prefix, (NEG_INF, NEG_INF))
                    e_non_blank = _log_add(e_non_blank, p_blank + token_logp)
                    next_beams[ext_prefix] = (e_blank, e_non_blank)
                else:
                    ext_prefix = prefix + (token,)
                    e_blank, e_non_blank = next_beams.get(ext_prefix, (NEG_INF, NEG_INF))
                    e_non_blank = _log_add(e_non_blank, prefix_total + token_logp)
                    next_beams[ext_prefix] = (e_blank, e_non_blank)

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _log_add(item[1][0], item[1][1]),
                reverse=True,
            )[:beam_size]
        )

    best_prefix, _ = max(
        beams.items(),
        key=lambda item: _log_add(item[1][0], item[1][1]),
    )
    return list(best_prefix)


def beam_search_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    beam_size: int = 10,
    blank_idx: int = 0,
    token_prune: Optional[int] = None,
) -> List[List[int]]:
    """
    Batch CTC prefix beam search.

    Args:
        log_probs: (B, T, V)
        lengths: (B,)
    Returns:
        List of token lists
    """
    decoded: List[List[int]] = []
    for i in range(log_probs.size(0)):
        length = int(lengths[i].item())
        if length <= 0:
            decoded.append([])
            continue
        seq = log_probs[i, :length]
        decoded.append(
            ctc_prefix_beam_search(
                seq,
                beam_size=beam_size,
                blank_idx=blank_idx,
                token_prune=token_prune,
            )
        )
    return decoded


def _labels_from_vocab(vocab: Sequence[str], blank_idx: int) -> List[str]:
    if blank_idx < 0 or blank_idx >= len(vocab):
        raise ValueError(
            f"blank_idx={blank_idx} is out of range for vocab size {len(vocab)}"
        )
    return ["" if idx == blank_idx else token for idx, token in enumerate(vocab)]


def build_lm_decoder(
    vocab: Sequence[str],
    lm_path: str,
    blank_idx: int = 0,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> Any:
    if not lm_path:
        raise ValueError("lm_path must be a non-empty path to a KenLM model file.")
    if build_ctcdecoder is None:
        raise RuntimeError(
            "pyctcdecode is not installed. Install with: pip install pyctcdecode kenlm"
        )

    model_path = Path(lm_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Language model file not found: {lm_path}")

    labels = _labels_from_vocab(vocab, blank_idx)
    return build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(model_path),
        alpha=alpha,
        beta=beta,
    )


def lm_beam_search_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    decoder: Any,
    beam_width: int = 100,
) -> List[str]:
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    decoded_text: List[str] = []
    for i in range(log_probs.size(0)):
        length = int(lengths[i].item())
        if length <= 0:
            decoded_text.append("")
            continue
        seq = log_probs[i, :length].detach().float().cpu().numpy()
        decoded_text.append(decoder.decode(seq, beam_width=beam_width))
    return decoded_text
