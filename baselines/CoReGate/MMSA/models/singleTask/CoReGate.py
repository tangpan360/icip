"""
CoReGate

Core idea (counterfactual contribution decomposition + reliability gating):
  - Predict text-only base y_T
  - Predict partial predictions y_TA (remove V), y_TV (remove A)
  - Define counterfactual (marginal) contributions:
      Δ_A  = y_TA  - y_T
      Δ_V  = y_TV  - y_T
  - Learn reliability gates r_A, r_V in (0,1) from (features + quality stats)
  - Final prediction:
      y = y_T + r_A * Δ_A + r_V * Δ_V
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..subNets import BertTextEncoder

__all__ = ["CoReGate"]


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    lengths: (B,) int
    returns: (B, max_len) bool mask (True for valid)
    """
    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths)
    lengths = lengths.to(dtype=torch.long)
    lengths = lengths.clamp(min=0, max=max_len)
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, L)
    return arange < lengths.unsqueeze(1)


def _sequence_valid_mask(x: torch.Tensor) -> torch.Tensor:
    """
    Heuristic valid mask for padded sequences.
    A timestep is valid if any feature dim is non-zero.
    """
    return (x.abs().sum(dim=-1) > 0).to(x.device)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: (B, L, D)
    mask: (B, L) bool (True for valid)
    returns: (B, D)
    """
    mask_f = mask.float().unsqueeze(-1)  # (B, L, 1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


def _attn_pool(
    q: torch.Tensor,
    kv: torch.Tensor,
    mask: torch.Tensor,
    proj_q: nn.Linear,
    proj_k: nn.Linear,
    dropout: nn.Dropout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dot-product attention pooling over a variable-length sequence.
    q: (B, Dq)
    kv: (B, L, Dkv)
    mask: (B, L) bool (True valid)
    returns:
      ctx: (B, Dkv)
      attn: (B, L)
    """
    qh = proj_q(q)  # (B, Dh)
    kh = proj_k(kv)  # (B, L, Dh)
    dh = qh.shape[-1]
    scores = (kh * qh.unsqueeze(1)).sum(dim=-1) / (float(dh) ** 0.5)  # (B, L)
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = dropout(attn)
    ctx = torch.bmm(attn.unsqueeze(1), kv).squeeze(1)  # (B, Dkv)
    return ctx, attn


def _quality_stats(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Simple per-sample quality stats for a modality.
      - mean L2 norm over valid steps
      - std  L2 norm over valid steps
      - valid ratio
    """
    l2 = torch.sqrt((x * x).sum(dim=-1).clamp_min(1e-12))  # (B, L)
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    mean = (l2 * mask_f).sum(dim=1) / denom
    var = ((l2 - mean.unsqueeze(1)) ** 2) * mask_f
    std = torch.sqrt(var.sum(dim=1) / denom).clamp_min(0.0)
    ratio = (mask_f.sum(dim=1) / mask_f.shape[1]).clamp(0.0, 1.0)
    return torch.stack([mean, std, ratio], dim=-1)  # (B, 3)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoReGate(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.output_dim = args.num_classes if args.train_mode == "classification" else 1

        # Text encoder (BERT/Roberta)
        self.text_encoder = BertTextEncoder(
            use_finetune=bool(getattr(args, "use_finetune", False)),
            transformers=str(getattr(args, "transformers", "bert")),
            pretrained=str(getattr(args, "pretrained", "bert-base-uncased")),
        )
        self.text_dim = 768

        # Audio/Video temporal encoders (simple BiLSTM)
        av_hidden = int(getattr(args, "av_hidden", 64))
        av_layers = int(getattr(args, "av_layers", 1))
        av_drop = float(getattr(args, "av_dropout", 0.1))

        self.audio_rnn = nn.LSTM(
            input_size=self.audio_in,
            hidden_size=av_hidden,
            num_layers=av_layers,
            dropout=0.0 if av_layers == 1 else av_drop,
            bidirectional=True,
            batch_first=True,
        )
        self.video_rnn = nn.LSTM(
            input_size=self.video_in,
            hidden_size=av_hidden,
            num_layers=av_layers,
            dropout=0.0 if av_layers == 1 else av_drop,
            bidirectional=True,
            batch_first=True,
        )
        self.av_out_dim = av_hidden * 2

        # Pooling
        self.use_attn_pool = bool(getattr(args, "use_attn_pool", True))
        attn_dim = int(getattr(args, "attn_dim", 128))
        attn_drop = float(getattr(args, "attn_dropout", 0.1))
        self._attn_dropout = nn.Dropout(p=attn_drop)
        self._proj_q_a = nn.Linear(self.text_dim, attn_dim)
        self._proj_k_a = nn.Linear(self.av_out_dim, attn_dim)
        self._proj_q_v = nn.Linear(self.text_dim, attn_dim)
        self._proj_k_v = nn.Linear(self.av_out_dim, attn_dim)

        self._av_dropout = nn.Dropout(p=float(getattr(args, "av_out_dropout", 0.1)))
        self._ln_a = nn.LayerNorm(self.av_out_dim)
        self._ln_v = nn.LayerNorm(self.av_out_dim)

        # Heads
        head_hidden = int(getattr(args, "head_hidden", 128))
        head_drop = float(getattr(args, "head_dropout", 0.1))

        self.head_t = _MLP(self.text_dim, head_hidden, self.output_dim, dropout=head_drop)
        self.head_ta = _MLP(self.text_dim + self.av_out_dim, head_hidden, self.output_dim, dropout=head_drop)
        self.head_tv = _MLP(self.text_dim + self.av_out_dim, head_hidden, self.output_dim, dropout=head_drop)

        # Gates
        self.gate_a = _MLP(self.text_dim + self.av_out_dim + 3, head_hidden, 1, dropout=head_drop)
        self.gate_v = _MLP(self.text_dim + self.av_out_dim + 3, head_hidden, 1, dropout=head_drop)

        # Ablation knobs (paper experiments)
        # - use_gating: if False, set r_A=r_V=1, i.e., no reliability gating.
        # - fusion_mode:
        #     "coregate" (default): y = y_T + r_A*Δ_A + r_V*Δ_V
        #     "t_only":            y = y_T
        self.use_gating = bool(getattr(args, "use_gating", True))
        self.fusion_mode = str(getattr(args, "fusion_mode", "coregate")).strip().lower()
        if self.fusion_mode not in ("coregate", "t_only"):
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

    def forward(self, text_x, audio_x, video_x):
        """
        audio_x / video_x can be:
        - Tensor: (B, L, D) or (B, 1, L, D) depending on preprocessing
        - Tuple: (Tensor, lengths) where lengths is (B,)
        """
        audio_lengths = None
        vision_lengths = None
        if isinstance(audio_x, (tuple, list)):
            audio_x, audio_lengths = audio_x[0], audio_x[1]
        if isinstance(video_x, (tuple, list)):
            video_x, vision_lengths = video_x[0], video_x[1]

        # Some pipelines may provide (B, 1, D) for normalized features.
        if torch.is_tensor(audio_x) and audio_x.dim() == 3 and audio_x.size(1) == 1:
            audio_x = audio_x.squeeze(1)
        if torch.is_tensor(video_x) and video_x.dim() == 3 and video_x.size(1) == 1:
            video_x = video_x.squeeze(1)

        # Text encoder: (B, L, 768) -> [CLS]
        t_states = self.text_encoder(text_x)
        h_t = t_states[:, 0, :]  # (B, 768)

        # Ensure A/V are sequences
        if audio_x.dim() == 2:
            audio_x = audio_x.unsqueeze(1)
        if video_x.dim() == 2:
            video_x = video_x.unsqueeze(1)

        # Masks: prefer true lengths from dataloader (unaligned); fall back to heuristic.
        if audio_lengths is not None:
            a_mask = _lengths_to_mask(torch.as_tensor(audio_lengths, device=audio_x.device), audio_x.shape[1])
        else:
            a_mask = _sequence_valid_mask(audio_x)
        if vision_lengths is not None:
            v_mask = _lengths_to_mask(torch.as_tensor(vision_lengths, device=video_x.device), video_x.shape[1])
        else:
            v_mask = _sequence_valid_mask(video_x)

        a_out, _ = self.audio_rnn(audio_x)
        v_out, _ = self.video_rnn(video_x)

        a_out = self._av_dropout(a_out)
        v_out = self._av_dropout(v_out)

        if self.use_attn_pool:
            h_a, attn_a = _attn_pool(h_t, a_out, a_mask, self._proj_q_a, self._proj_k_a, self._attn_dropout)
            h_v, attn_v = _attn_pool(h_t, v_out, v_mask, self._proj_q_v, self._proj_k_v, self._attn_dropout)
        else:
            attn_a, attn_v = None, None
            h_a = _masked_mean(a_out, a_mask)
            h_v = _masked_mean(v_out, v_mask)

        h_a = self._ln_a(h_a)
        h_v = self._ln_v(h_v)

        q_a = _quality_stats(audio_x, a_mask)  # (B, 3)
        q_v = _quality_stats(video_x, v_mask)  # (B, 3)

        # Counterfactual predictions
        y_t = self.head_t(h_t)
        y_ta = self.head_ta(torch.cat([h_t, h_a], dim=-1))
        y_tv = self.head_tv(torch.cat([h_t, h_v], dim=-1))

        # Marginal contributions
        d_a = y_ta - y_t
        d_v = y_tv - y_t

        # Reliability gates
        r_a = torch.sigmoid(self.gate_a(torch.cat([h_t, h_a, q_a], dim=-1)))  # (B,1)
        r_v = torch.sigmoid(self.gate_v(torch.cat([h_t, h_v, q_v], dim=-1)))  # (B,1)

        # Ablation: no gating
        if not self.use_gating:
            r_a = torch.ones_like(r_a)
            r_v = torch.ones_like(r_v)

        # Output selection (ablation)
        if self.fusion_mode == "t_only":
            y = y_t
        else:
            y = y_t + r_a * d_a + r_v * d_v

        return {
            "Feature_t": h_t,
            "Feature_a": h_a,
            "Feature_v": h_v,
            "Feature_f": torch.cat([h_t, h_a, h_v], dim=-1),
            "M": y,
            # counterfactual heads
            "y_t": y_t,
            "y_ta": y_ta,
            "y_tv": y_tv,
            # contributions / gates
            "delta_a": d_a,
            "delta_v": d_v,
            "r_a": r_a,
            "r_v": r_v,
            "q_a": q_a,
            "q_v": q_v,
            "attn_a": attn_a,
            "attn_v": attn_v,
        }

