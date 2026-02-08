"""Decoding algorithms for CTC and RNN-T models.

CTC:
  - ctc_prefix_beam_search / beam_search_decode (pure Python)
  - LM-fused CTC decoding via pyctcdecode (optional)

RNN-T:
  - rnnt_greedy_decode   (fast, used during training eval)
  - rnnt_beam_search     (higher quality, used for final eval)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from pyctcdecode import build_ctcdecoder
except ImportError:
    build_ctcdecoder = None


NEG_INF = float("-inf")


# ===================================================================
# CTC decoding
# ===================================================================


def _log_add(a: float, b: float) -> float:
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
    """CTC prefix beam search on a single sequence.

    Args:
        log_probs: (T, V) log probabilities
        beam_size: number of active beams
        blank_idx: CTC blank token index
        token_prune: optional per-frame top-k pruning
    Returns:
        Best decoded token sequence
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
                    nb, nnb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    nb = _log_add(nb, prefix_total + token_logp)
                    next_beams[prefix] = (nb, nnb)
                    continue

                if token == last_token:
                    nb, nnb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    nnb = _log_add(nnb, p_non_blank + token_logp)
                    next_beams[prefix] = (nb, nnb)

                    ext = prefix + (token,)
                    eb, enb = next_beams.get(ext, (NEG_INF, NEG_INF))
                    enb = _log_add(enb, p_blank + token_logp)
                    next_beams[ext] = (eb, enb)
                else:
                    ext = prefix + (token,)
                    eb, enb = next_beams.get(ext, (NEG_INF, NEG_INF))
                    enb = _log_add(enb, prefix_total + token_logp)
                    next_beams[ext] = (eb, enb)

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda kv: _log_add(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_size]
        )

    best, _ = max(beams.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]))
    return list(best)


def beam_search_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    beam_size: int = 10,
    blank_idx: int = 0,
    token_prune: Optional[int] = None,
) -> List[List[int]]:
    """Batch CTC prefix beam search."""
    decoded: List[List[int]] = []
    for i in range(log_probs.size(0)):
        length = int(lengths[i].item())
        if length <= 0:
            decoded.append([])
            continue
        decoded.append(
            ctc_prefix_beam_search(
                log_probs[i, :length],
                beam_size=beam_size,
                blank_idx=blank_idx,
                token_prune=token_prune,
            )
        )
    return decoded


# ---------------------------------------------------------------------------
# CTC + LM (pyctcdecode)
# ---------------------------------------------------------------------------


def _labels_from_vocab(vocab: Sequence[str], blank_idx: int) -> List[str]:
    if blank_idx < 0 or blank_idx >= len(vocab):
        raise ValueError(f"blank_idx={blank_idx} out of range for vocab size {len(vocab)}")
    return ["" if idx == blank_idx else token for idx, token in enumerate(vocab)]


def build_lm_decoder(
    vocab: Sequence[str],
    lm_path: str,
    blank_idx: int = 0,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> Any:
    if not lm_path:
        raise ValueError("lm_path must be a non-empty path.")
    if build_ctcdecoder is None:
        raise RuntimeError("pyctcdecode is not installed. pip install pyctcdecode kenlm")
    if not Path(lm_path).exists():
        raise FileNotFoundError(f"Language model not found: {lm_path}")

    labels = _labels_from_vocab(vocab, blank_idx)
    return build_ctcdecoder(labels=labels, kenlm_model_path=str(lm_path), alpha=alpha, beta=beta)


def lm_beam_search_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    decoder: Any,
    beam_width: int = 100,
) -> List[str]:
    decoded_text: List[str] = []
    for i in range(log_probs.size(0)):
        length = int(lengths[i].item())
        if length <= 0:
            decoded_text.append("")
            continue
        seq = log_probs[i, :length].detach().float().cpu().numpy()
        decoded_text.append(decoder.decode(seq, beam_width=beam_width))
    return decoded_text


# ===================================================================
# RNN-T decoding
# ===================================================================


@torch.no_grad()
def rnnt_greedy_decode(
    model,
    enc_out: torch.Tensor,
    enc_lengths: torch.Tensor,
    blank_idx: int = 0,
    max_symbols_per_step: int = 10,
) -> List[List[int]]:
    """RNN-T greedy decoding. Fast, suitable for eval during training.

    Args:
        model: ConformerTransducer (or any model with .predictor and .joint)
        enc_out: (B, T, D_enc) encoder output
        enc_lengths: (B,) valid lengths
        blank_idx: blank token index
        max_symbols_per_step: cap on label emissions per encoder frame
    Returns:
        List of decoded token-ID lists (one per batch item)
    """
    B = enc_out.size(0)
    device = enc_out.device
    results: List[List[int]] = []

    for b in range(B):
        T = int(enc_lengths[b].item())
        hyp: List[int] = []
        state = None

        # Initial predictor step: feed blank / SOS
        pred_input = torch.tensor([[blank_idx]], device=device, dtype=torch.long)
        pred_out, state = model.predictor(pred_input, state)
        pred_vec = pred_out[0, 0]  # (D_pred,)

        for t in range(T):
            enc_vec = enc_out[b, t]  # (D_enc,)

            for _ in range(max_symbols_per_step):
                logits = model.joint.forward_step(
                    enc_vec.unsqueeze(0), pred_vec.unsqueeze(0)
                )  # (1, V)
                token_id = int(logits.argmax(dim=-1).item())

                if token_id == blank_idx:
                    break

                hyp.append(token_id)
                pred_input = torch.tensor([[token_id]], device=device, dtype=torch.long)
                pred_out, state = model.predictor(pred_input, state)
                pred_vec = pred_out[0, 0]

        results.append(hyp)

    return results


# ---------------------------------------------------------------------------
# RNN-T Beam Search
# ---------------------------------------------------------------------------

class _Beam:
    """Internal beam hypothesis container."""

    __slots__ = ("hyp", "score", "state", "pred_vec")

    def __init__(self, hyp: List[int], score: float, state, pred_vec: torch.Tensor):
        self.hyp = hyp
        self.score = score
        self.state = state
        self.pred_vec = pred_vec


def _clone_state(state):
    """Deep-clone LSTM hidden state tuple."""
    if state is None:
        return None
    h, c = state
    return (h.clone(), c.clone())


@torch.no_grad()
def rnnt_beam_search(
    model,
    enc_out: torch.Tensor,
    enc_lengths: torch.Tensor,
    blank_idx: int = 0,
    beam_size: int = 10,
    max_symbols_per_step: int = 10,
    top_k_tokens: int = 10,
    lm=None,
    lm_weight: float = 0.0,
) -> List[List[int]]:
    """RNN-T beam search decoding with optional shallow-fusion LM.

    At each encoder frame, beams can emit multiple labels (up to
    max_symbols_per_step) before consuming the blank and advancing.

    Args:
        model: ConformerTransducer
        enc_out: (B, T, D_enc) encoder output
        enc_lengths: (B,)
        blank_idx: blank index
        beam_size: number of active hypotheses
        max_symbols_per_step: max label emissions per encoder frame
        top_k_tokens: only consider top-k non-blank tokens per expansion
        lm: optional language model with .score(token_ids) -> float
        lm_weight: weight for LM shallow fusion (0 disables)
    Returns:
        List of best hypothesis token-ID lists
    """
    B = enc_out.size(0)
    device = enc_out.device
    all_results: List[List[int]] = []

    for b in range(B):
        T = int(enc_lengths[b].item())

        # Initialize single beam with blank SOS
        pred_input = torch.tensor([[blank_idx]], device=device, dtype=torch.long)
        pred_out, init_state = model.predictor(pred_input)
        init_pred_vec = pred_out[0, 0]

        beams = [_Beam(hyp=[], score=0.0, state=init_state, pred_vec=init_pred_vec)]

        for t in range(T):
            enc_vec = enc_out[b, t]  # (D_enc,)

            # Beams that already emitted blank for this step (done)
            finished_step: List[_Beam] = []
            # Active beams that can still emit tokens at this step
            active = list(beams)

            for _sym_iter in range(max_symbols_per_step):
                if not active:
                    break

                next_active: List[_Beam] = []

                for beam in active:
                    logits = model.joint.forward_step(
                        enc_vec.unsqueeze(0), beam.pred_vec.unsqueeze(0)
                    )  # (1, V)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)

                    # Blank transition: move to next time step
                    blank_score = beam.score + float(log_probs[blank_idx].item())
                    finished_step.append(
                        _Beam(
                            hyp=beam.hyp,
                            score=blank_score,
                            state=beam.state,
                            pred_vec=beam.pred_vec,
                        )
                    )

                    # Top-k non-blank token transitions
                    # Mask blank before top-k selection
                    non_blank_probs = log_probs.clone()
                    non_blank_probs[blank_idx] = NEG_INF
                    k = min(top_k_tokens, non_blank_probs.size(-1))
                    top_vals, top_ids = non_blank_probs.topk(k)

                    for i in range(k):
                        token_id = int(top_ids[i].item())
                        token_score = float(top_vals[i].item())

                        if token_score == NEG_INF:
                            continue

                        new_score = beam.score + token_score

                        # Optional LM shallow fusion
                        if lm is not None and lm_weight > 0:
                            new_hyp_ids = beam.hyp + [token_id]
                            new_score += lm_weight * lm.score(new_hyp_ids)

                        # Run predictor for new token
                        new_state = _clone_state(beam.state)
                        pred_input = torch.tensor([[token_id]], device=device, dtype=torch.long)
                        pred_out, new_state = model.predictor(pred_input, new_state)
                        new_pred_vec = pred_out[0, 0]

                        next_active.append(
                            _Beam(
                                hyp=beam.hyp + [token_id],
                                score=new_score,
                                state=new_state,
                                pred_vec=new_pred_vec,
                            )
                        )

                # Prune active beams
                next_active.sort(key=lambda x: x.score, reverse=True)
                active = next_active[:beam_size]

            # Combine finished + any remaining active, prune to beam_size
            all_candidates = finished_step + active
            all_candidates.sort(key=lambda x: x.score, reverse=True)
            beams = all_candidates[:beam_size]

        best = beams[0] if beams else _Beam([], 0.0, None, init_pred_vec)
        all_results.append(best.hyp)

    return all_results