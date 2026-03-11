"""
AISTransformer: transformer encoder for next-step displacement prediction.

pred_mode="causal"  (default):
    Causal (triangular) attention mask — position t cannot attend to t+1, t+2...
    Output at position t predicts the displacement to position t+1.
    At inference, read off the last output to get the 1-step-ahead prediction.

pred_mode="single":
    No mask. Reads the full window bidirectionally.
    Uses mean pooling over the sequence to produce one prediction vector.

Vessel type conditioning:
    If num_vessel_types > 0, a learned embedding for each AIS vessel type code is
    added to the model's hidden state after input projection (broadcast across all
    timesteps, like BERT segment embeddings). This lets the model learn
    type-specific motion patterns (tanker vs passenger vs cargo).
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))   # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(0)])


class AISTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dim_feedforward: int = None,
                 dropout: float = 0.1, pred_mode: str = "causal",
                 num_vessel_types: int = 0, vessel_type_embed_dim: int = 8):
        super().__init__()
        assert d_model % nhead == 0
        assert pred_mode in ("causal", "single")
        self.pred_mode = pred_mode
        ffn = dim_feedforward or 4 * d_model

        self.input_proj  = nn.Linear(input_dim, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=ffn,
                dropout=dropout, batch_first=False,
            ),
            num_layers=num_layers,
        )
        self.output_proj = nn.Linear(d_model, 2)

        # Optional vessel type conditioning (BERT-style segment embedding)
        if num_vessel_types > 0:
            self.vessel_type_emb  = nn.Embedding(num_vessel_types, vessel_type_embed_dim)
            self.vessel_type_proj = nn.Linear(vessel_type_embed_dim, d_model)
            nn.init.normal_(self.vessel_type_emb.weight, std=0.02)
            nn.init.xavier_uniform_(self.vessel_type_proj.weight)
            nn.init.zeros_(self.vessel_type_proj.bias)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.uniform_(self.output_proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.output_proj.bias)

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x: torch.Tensor,
                gap_mask: torch.Tensor = None,
                vessel_type: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x           : (seq_len, B, input_dim)
            gap_mask    : (B, seq_len) bool — True = masked position (large time gap)
            vessel_type : (B,) int64 — vessel type vocab index per sample
        Returns:
            causal : (seq_len, B, 2)  predicted displacement at each step
            single : (B, 2)           single displacement prediction
        """
        x = self.input_proj(x)   # (seq_len, B, d_model)

        # Vessel type conditioning: add per-vessel bias, broadcast over seq_len
        if hasattr(self, "vessel_type_emb") and vessel_type is not None:
            type_emb = self.vessel_type_proj(
                self.vessel_type_emb(vessel_type))  # (B, d_model)
            x = x + type_emb.unsqueeze(0)           # (seq_len, B, d_model)

        x = self.pos_enc(x)

        if self.pred_mode == "causal":
            mask = self._causal_mask(x.size(0), x.device)
            x = self.transformer(x, mask=mask, src_key_padding_mask=gap_mask)
            return self.output_proj(x)        # (seq_len, B, 2)
        else:
            x = self.transformer(x, src_key_padding_mask=gap_mask)
            x = x.mean(dim=0)                # (B, d_model)
            return self.output_proj(x)        # (B, 2)
