#!/usr/bin/env python3
"""
train_ais_transformer.py -- AIS Vessel Trajectory Prediction

TASK: Given a sequence of past positions, predict the next position.

Two modes controlled by --pred_mode:

  "causal" (default, recommended):
      Train seq2seq with a CAUSAL MASK. At each timestep t the model
      only attends to positions 0..t (no future leakage). Output at
      position t predicts position t+1. At inference, feed the window
      and read off the last output as the 1-step-ahead prediction.
      This is the correct way to use a transformer encoder for this task.

  "single":
      Each sample = (context window of length seq_len, single next position).
      The model reads the full window bidirectionally and outputs one prediction.
      No leakage. Slightly simpler but wastes the intermediate supervision signal.

The previous "bidirectional seq2seq" approach was wrong: without a causal mask
the model at position t can attend to future positions t+1..T in the input,
meaning it can "cheat" by copying nearby future values. This causes it to learn
a "copy-input" strategy that looks good in training MSE but fails at inference
when future positions are not available.

Input features per timestep:
  [lon_norm, lat_norm, log_dt_norm]          (3-dim, default)
  [lon_norm, lat_norm, log_dt_norm, dlon_norm, dlat_norm]  (5-dim, --use_velocity)

Target: next [lon_norm, lat_norm]

Usage:
    python train_ais_transformer.py --csv data/processed/ais.csv
    python train_ais_transformer.py --csv data/processed/ais.csv \\
        --pred_mode single --seq_len 30 --d_model 256 --num_layers 4
"""

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility + device
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Scalers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scaler:
    mean: np.ndarray
    std:  np.ndarray

    @staticmethod
    def fit(x: np.ndarray) -> "Scaler":
        mean = x.mean(axis=0).astype(np.float32)
        std  = x.std(axis=0).astype(np.float32)
        std  = np.where(std < 1e-6, 1.0, std)
        return Scaler(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


@dataclass
class Scalers:
    pos:   Scaler   # [lon, lat]
    logdt: Scaler   # log1p(dt_seconds)

    @staticmethod
    def fit(tracks: Dict[str, np.ndarray]) -> "Scalers":
        all_pos, all_logdt = [], []
        for pts in tracks.values():
            all_pos.append(pts[:, :2])
            all_logdt.append(np.log1p(pts[:, 2]).reshape(-1, 1))
        return Scalers(
            pos   = Scaler.fit(np.concatenate(all_pos,   axis=0)),
            logdt = Scaler.fit(np.concatenate(all_logdt, axis=0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tracks(
    csv_path: str,
    id_col:   str = "MMSI",
    time_col: str = "BaseDateTime",
    lat_col:  str = "LAT",
    lon_col:  str = "LON",
) -> Dict[str, np.ndarray]:
    """Returns dict: mmsi -> float32 (T, 3) = [lon, lat, dt_seconds]"""
    df = pd.read_csv(csv_path)
    missing = set([id_col, time_col, lat_col, lon_col]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[id_col, time_col, lat_col, lon_col])
    df = df.sort_values([id_col, time_col])

    tracks: Dict[str, np.ndarray] = {}
    for v_id, g in df.groupby(id_col):
        lon = g[lon_col].to_numpy(dtype=np.float32)
        lat = g[lat_col].to_numpy(dtype=np.float32)
        ts  = g[time_col].values.astype("datetime64[s]").astype(np.float64)
        dt  = np.concatenate([[0.0], np.diff(ts)]).astype(np.float32)
        pts = np.stack([lon, lat, dt], axis=1)
        if pts.shape[0] >= 5:
            tracks[str(v_id)] = pts

    lengths = [v.shape[0] for v in tracks.values()]
    print(f"Loaded {len(tracks):,} vessels")
    print(f"  Track length -- min:{min(lengths)}  "
          f"median:{np.median(lengths):.0f}  max:{max(lengths)}")
    return tracks


def train_val_split(tracks, val_frac, seed):
    ids = list(tracks.keys())
    random.Random(seed).shuffle(ids)
    if len(ids) <= 1 or val_frac <= 0:
        return tracks, {}
    n_val   = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    return (
        {k: v for k, v in tracks.items() if k not in val_ids},
        {k: v for k, v in tracks.items() if k in val_ids},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_input(pts: np.ndarray, scalers: Scalers, use_velocity: bool) -> np.ndarray:
    """
    Build normalised input features from a (T, 3) trajectory window.
    Returns (T, 3) or (T, 5).
    """
    pos_norm   = scalers.pos.transform(pts[:, :2])
    logdt_norm = scalers.logdt.transform(np.log1p(pts[:, 2]).reshape(-1, 1))
    x = np.concatenate([pos_norm, logdt_norm], axis=1)
    if use_velocity:
        dpos     = np.diff(pts[:, :2], axis=0, prepend=pts[0:1, :2])
        vel_norm = (dpos / scalers.pos.std).astype(np.float32)
        x = np.concatenate([x, vel_norm], axis=1)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Dataset -- CAUSAL mode
# ─────────────────────────────────────────────────────────────────────────────

class CausalDataset(Dataset):
    """
    seq2seq with causal mask.

    x : (seq_len, input_dim)   positions 0..seq_len-1
    y : (seq_len, 2)           targets = positions 1..seq_len (next pos at each step)

    At training, a causal mask ensures position t only attends to 0..t.
    At inference, the last output (y[-1]) is the 1-step-ahead prediction
    given the full context window, with no future leakage.
    """

    def __init__(self, tracks, scalers, seq_len, stride=1,
                 max_windows_per_track=None, use_velocity=False,
                 max_gap_sec=600):
        self.tracks       = tracks
        self.scalers      = scalers
        self.seq_len      = seq_len
        self.use_velocity = use_velocity
        self.max_gap_sec  = max_gap_sec
        self.index: List[Tuple[str, int]] = []

        for v_id, pts in tracks.items():
            T = pts.shape[0]
            if T < seq_len + 1:
                continue
            starts = list(range(0, T - seq_len, stride))
            if max_windows_per_track and len(starts) > max_windows_per_track:
                starts = sorted(random.sample(starts, max_windows_per_track))
            for s in starts:
                self.index.append((v_id, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        v_id, s = self.index[idx]
        pts = self.tracks[v_id][s : s + self.seq_len + 1]  # (seq_len+1, 3)

        x = make_input(pts[:-1], self.scalers, self.use_velocity)  # (seq_len, D)
        y = self.scalers.pos.transform(pts[1:, :2])                 # (seq_len, 2)

        # gap_mask[t] = True  →  position t follows a large time gap.
        # Passed as src_key_padding_mask so the model cannot attend TO it.
        dt = pts[:-1, 2]  # dt_seconds for each input position
        gap_mask = torch.tensor(dt > self.max_gap_sec, dtype=torch.bool)  # (seq_len,)

        return torch.tensor(x), torch.tensor(y), gap_mask


# ─────────────────────────────────────────────────────────────────────────────
# Dataset -- SINGLE mode
# ─────────────────────────────────────────────────────────────────────────────

class SingleStepDataset(Dataset):
    """
    Predict one step ahead from a context window.

    x : (seq_len, input_dim)  positions 0..seq_len-1 (full context)
    y : (2,)                  target = position seq_len (next after window)

    The model attends bidirectionally over x and outputs a single prediction.
    No causal mask needed. No leakage because y is not in x.
    """

    def __init__(self, tracks, scalers, seq_len, stride=1,
                 max_windows_per_track=None, use_velocity=False,
                 max_gap_sec=600):
        self.tracks       = tracks
        self.scalers      = scalers
        self.seq_len      = seq_len
        self.use_velocity = use_velocity
        self.max_gap_sec  = max_gap_sec
        self.index: List[Tuple[str, int]] = []

        for v_id, pts in tracks.items():
            T = pts.shape[0]
            if T < seq_len + 1:
                continue
            starts = list(range(0, T - seq_len, stride))
            if max_windows_per_track and len(starts) > max_windows_per_track:
                starts = sorted(random.sample(starts, max_windows_per_track))
            for s in starts:
                self.index.append((v_id, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        v_id, s = self.index[idx]
        pts = self.tracks[v_id][s : s + self.seq_len + 1]

        x = make_input(pts[:-1], self.scalers, self.use_velocity)  # (seq_len, D)
        y = self.scalers.pos.transform(pts[-1:, :2])[0]             # (2,)

        dt = pts[:-1, 2]
        gap_mask = torch.tensor(dt > self.max_gap_sec, dtype=torch.bool)  # (seq_len,)

        return torch.tensor(x), torch.tensor(y), gap_mask


def causal_collate(batch):
    xs, ys, gms = zip(*batch)
    return (
        torch.stack(xs).transpose(0, 1),   # (seq_len, B, D)
        torch.stack(ys).transpose(0, 1),   # (seq_len, B, 2)
        torch.stack(gms),                   # (B, seq_len)
    )


def single_collate(batch):
    xs, ys, gms = zip(*batch)
    return (
        torch.stack(xs).transpose(0, 1),   # (seq_len, B, D)
        torch.stack(ys),                    # (B, 2)
        torch.stack(gms),                   # (B, seq_len)
    )


def get_loader(tracks, scalers, seq_len, batch_size, pred_mode="causal",
               stride=1, max_windows_per_track=None,
               shuffle=True, drop_last=True,
               use_velocity=False, num_workers=0, max_gap_sec=600):
    if pred_mode == "causal":
        dataset  = CausalDataset(tracks, scalers, seq_len, stride,
                                 max_windows_per_track, use_velocity,
                                 max_gap_sec=max_gap_sec)
        collate  = causal_collate
    else:
        dataset  = SingleStepDataset(tracks, scalers, seq_len, stride,
                                     max_windows_per_track, use_velocity,
                                     max_gap_sec=max_gap_sec)
        collate  = single_collate

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])


class AISTransformer(nn.Module):
    """
    Transformer encoder for next-position prediction.

    pred_mode="causal":
        Uses a causal (triangular) attention mask. Output at position t
        predicts position t+1. At inference, the last output is used.

    pred_mode="single":
        No mask. Reads full window bidirectionally. Uses mean pooling
        over the sequence to produce a single prediction vector.
    """

    def __init__(self, input_dim, d_model=128, nhead=8,
                 num_layers=2, dim_feedforward=None, dropout=0.1,
                 pred_mode="causal"):
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
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.uniform_(self.output_proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.output_proj.bias)

    def _causal_mask(self, sz, device):
        """Upper-triangular mask: position t cannot attend to t+1, t+2, ..."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x, gap_mask=None):
        """
        Args:
            x        : (seq_len, B, input_dim)
            gap_mask : (B, seq_len) bool — True = position follows a large time
                       gap and should be ignored as a key (src_key_padding_mask).
        Returns:
            causal mode : (seq_len, B, 2)  prediction at each step
            single mode : (B, 2)           single prediction from full context
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)

        if self.pred_mode == "causal":
            mask = self._causal_mask(x.size(0), x.device)
            x = self.transformer(x, mask=mask, src_key_padding_mask=gap_mask)
            return self.output_proj(x)           # (seq_len, B, 2)
        else:
            x = self.transformer(x, src_key_padding_mask=gap_mask)
            x = x.mean(dim=0)                    # (B, d_model)  mean pool
            return self.output_proj(x)           # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, scalers, pred_mode):
    """
    Returns mse (normalised), ADE and FDE in metres.

    For causal mode: ADE averages over all seq positions, FDE uses last.
    For single mode: ADE == FDE == the single-step error.
    """
    model.eval()
    mse_fn = nn.MSELoss()
    total_mse, ade_sum, fde_sum, n = 0.0, 0.0, 0.0, 0

    for x, y, gap_mask in loader:
        x, y = x.to(device), y.to(device)
        gap_mask = gap_mask.to(device)
        pred = model(x, gap_mask=gap_mask)

        total_mse += mse_fn(pred, y).item()
        n += 1

        if pred_mode == "causal":
            # pred: (seq_len, B, 2),  y: (seq_len, B, 2)
            pred_geo = scalers.pos.inverse(
                pred.cpu().numpy().reshape(-1, 2)).reshape(pred.shape)
            y_geo = scalers.pos.inverse(
                y.cpu().numpy().reshape(-1, 2)).reshape(y.shape)

            dist = haversine_meters(y_geo[:, :, 1], y_geo[:, :, 0],
                                    pred_geo[:, :, 1], pred_geo[:, :, 0])
            ade_sum += float(dist.mean())
            fde_sum += float(dist[-1].mean())
        else:
            # pred: (B, 2),  y: (B, 2)
            pred_geo = scalers.pos.inverse(pred.cpu().numpy())
            y_geo    = scalers.pos.inverse(y.cpu().numpy())
            dist = haversine_meters(y_geo[:, 1], y_geo[:, 0],
                                    pred_geo[:, 1], pred_geo[:, 0])
            ade_sum += float(dist.mean())
            fde_sum += float(dist.mean())

    if n == 0:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}
    return {"mse": total_mse / n, "ade_m": ade_sum / n, "fde_m": fde_sum / n}


@torch.no_grad()
def evaluate_constant_velocity_baseline(loader, device, scalers, pred_mode):
    """
    Baseline: next_pos = current_pos + (current_pos - prev_pos)
    Evaluated in the same way as the model.
    """
    mse_fn = nn.MSELoss()
    total_mse, ade_sum, fde_sum, n = 0.0, 0.0, 0.0, 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pos = x[:, :, :2]   # (seq_len, B, 2) normalised lon/lat

        if pred_mode == "causal":
            vel  = torch.diff(pos, dim=0, prepend=pos[:1])
            pred = pos + vel           # (seq_len, B, 2) extrapolated position

            total_mse += mse_fn(pred, y).item()
            n += 1

            pred_geo = scalers.pos.inverse(
                pred.cpu().numpy().reshape(-1, 2)).reshape(pred.shape)
            y_geo = scalers.pos.inverse(
                y.cpu().numpy().reshape(-1, 2)).reshape(y.shape)
            dist = haversine_meters(y_geo[:, :, 1], y_geo[:, :, 0],
                                    pred_geo[:, :, 1], pred_geo[:, :, 0])
            ade_sum += float(dist.mean())
            fde_sum += float(dist[-1].mean())
        else:
            # For single mode: extrapolate from last two positions in window
            vel  = pos[-1] - pos[-2]          # (B, 2)
            pred = pos[-1] + vel              # (B, 2)

            total_mse += mse_fn(pred, y).item()
            n += 1

            pred_geo = scalers.pos.inverse(pred.cpu().numpy())
            y_geo    = scalers.pos.inverse(y.cpu().numpy())
            dist = haversine_meters(y_geo[:, 1], y_geo[:, 0],
                                    pred_geo[:, 1], pred_geo[:, 0])
            ade_sum += float(dist.mean())
            fde_sum += float(dist.mean())

    if n == 0:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}
    return {"mse": total_mse / n, "ade_m": ade_sum / n, "fde_m": fde_sum / n}


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sanity_check(model, loader, device, scalers, pred_mode):
    """Print a few raw predictions vs targets to verify sensible output."""
    model.eval()
    x, y, gap_mask = next(iter(loader))
    x, y = x.to(device), y.to(device)
    gap_mask = gap_mask.to(device)
    pred = model(x, gap_mask=gap_mask)

    if pred_mode == "causal":
        # Use last timestep
        p_batch = pred[-1]   # (B, 2)
        t_batch = y[-1]
    else:
        p_batch = pred       # (B, 2)
        t_batch = y

    for i in range(min(3, p_batch.shape[0])):
        p = scalers.pos.inverse(p_batch[i].cpu().numpy().reshape(1, 2))[0]
        t = scalers.pos.inverse(t_batch[i].cpu().numpy().reshape(1, 2))[0]
        err = haversine_meters(t[1], t[0], p[1], p[0])
        print(f"    sample {i}: pred ({p[0]:.3f}°, {p[1]:.3f}°)  "
              f"true ({t[0]:.3f}°, {t[1]:.3f}°)  err {err:.0f}m")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler,
                    device, grad_clip, epoch, pred_mode, log_every=50):
    model.train()
    mse_fn = nn.MSELoss()
    running, total, n = 0.0, 0.0, 0

    for i, (x, y, gap_mask) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        gap_mask = gap_mask.to(device)
        optimizer.zero_grad()
        pred = model(x, gap_mask=gap_mask)
        loss = mse_fn(pred, y)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        running += loss.item()
        total   += loss.item()
        n       += 1

        if (i + 1) % log_every == 0:
            print(f"  epoch {epoch:3d} | step {i+1:5d}/{len(loader)} "
                  f"| mse {running/log_every:.6f} "
                  f"| grad {grad_norm:.3f} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e}")
            running = 0.0

    return total / max(n, 1)


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--csv",       required=True)
    parser.add_argument("--out_dir",   default="runs/ais_transformer")
    parser.add_argument("--id_col",    default="MMSI")
    parser.add_argument("--time_col",  default="BaseDateTime")
    parser.add_argument("--lat_col",   default="LAT")
    parser.add_argument("--lon_col",   default="LON")

    parser.add_argument("--pred_mode",  default="causal",
                        choices=["causal", "single"],
                        help="causal=seq2seq with causal mask (recommended). "
                             "single=predict one step from full bidirectional context.")
    parser.add_argument("--seq_len",               type=int,   default=20)
    parser.add_argument("--stride",                type=int,   default=5)
    parser.add_argument("--max_windows_per_track", type=int,   default=None)
    parser.add_argument("--max_gap_sec",           type=float, default=600.0,
                        help="Time gaps > this (seconds) are masked as padding keys "
                             "so the model cannot attend across the gap (default: 600s = 10min)")
    parser.add_argument("--use_velocity",          action="store_true")
    parser.add_argument("--val_frac",              type=float, default=0.1)
    parser.add_argument("--seed",                  type=int,   default=42)

    parser.add_argument("--d_model",         type=int,   default=128)
    parser.add_argument("--nhead",           type=int,   default=8)
    parser.add_argument("--num_layers",      type=int,   default=2)
    parser.add_argument("--dim_feedforward", type=int,   default=None)
    parser.add_argument("--dropout",         type=float, default=0.1)

    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--warmup_frac",  type=float, default=0.05)
    parser.add_argument("--num_workers",  type=int,   default=0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  AIS TRANSFORMER -- TRAINING")
    print("=" * 65)
    print(f"  Device    : {device}")
    print(f"  Pred mode : {args.pred_mode}")
    print(f"  Out dir   : {args.out_dir}\n")

    tracks = load_tracks(args.csv, args.id_col, args.time_col,
                         args.lat_col, args.lon_col)
    if not tracks:
        raise RuntimeError("No valid tracks loaded.")

    train_tracks, val_tracks = train_val_split(tracks, args.val_frac, args.seed)
    print(f"  Train: {len(train_tracks):,} vessels  |  Val: {len(val_tracks):,}\n")

    scalers = Scalers.fit(train_tracks)
    print(f"  Position scaler -- mean: {scalers.pos.mean}  std: {scalers.pos.std}")
    print(f"  log(dt) scaler  -- mean: {scalers.logdt.mean.item():.4f}  "
          f"std: {scalers.logdt.std.item():.4f}\n")

    input_dim = 5 if args.use_velocity else 3
    print(f"  Input  : {input_dim}D "
          f"{'[lon,lat,log_dt,dlon,dlat]' if args.use_velocity else '[lon,lat,log_dt]'}")
    print(f"  Target : next [lon_norm, lat_norm]")
    print(f"  Mode   : {args.pred_mode}\n")

    train_loader = get_loader(
        train_tracks, scalers, args.seq_len, args.batch_size,
        pred_mode=args.pred_mode, stride=args.stride,
        max_windows_per_track=args.max_windows_per_track,
        shuffle=True, drop_last=True, use_velocity=args.use_velocity,
        num_workers=args.num_workers, max_gap_sec=args.max_gap_sec,
    )
    val_loader = get_loader(
        val_tracks, scalers, args.seq_len, args.batch_size,
        pred_mode=args.pred_mode, stride=args.stride,
        max_windows_per_track=args.max_windows_per_track,
        shuffle=False, drop_last=False, use_velocity=args.use_velocity,
        num_workers=args.num_workers, max_gap_sec=args.max_gap_sec,
    ) if val_tracks else None

    print(f"  Train batches : {len(train_loader):,}")
    if val_loader:
        print(f"  Val batches   : {len(val_loader):,}")

    model = AISTransformer(
        input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, pred_mode=args.pred_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Params  : {n_params:,}")
    print(f"  d_model={args.d_model}  nhead={args.nhead}  "
          f"layers={args.num_layers}  ffn={args.dim_feedforward or 4*args.d_model}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)
    print(f"  Steps   : {total_steps:,} total | {warmup_steps:,} warmup\n")

    if val_loader:
        print("─" * 65)
        bl = evaluate_constant_velocity_baseline(
            val_loader, device, scalers, args.pred_mode)
        print(f"  Const-vel baseline -- "
              f"MSE {bl['mse']:.6f}  ADE {bl['ade_m']:.1f}m  FDE {bl['fde_m']:.1f}m")
        print(f"  Beat this or something is wrong.")
        print("─" * 65 + "\n")

    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.grad_clip, epoch, args.pred_mode,
        )

        log = f"Epoch {epoch:3d}/{args.epochs} | train mse {train_mse:.6f}"

        if val_loader:
            metrics = evaluate(model, val_loader, device, scalers, args.pred_mode)
            log += (f" | val mse {metrics['mse']:.6f} | "
                    f"ADE {metrics['ade_m']:.1f}m | FDE {metrics['fde_m']:.1f}m")

            if metrics["mse"] < best_val_mse:
                best_val_mse = metrics["mse"]
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "pos_mean": scalers.pos.mean, "pos_std": scalers.pos.std,
                    "logdt_mean": scalers.logdt.mean, "logdt_std": scalers.logdt.std,
                    "metrics": metrics, "args": vars(args),
                }, os.path.join(args.out_dir, "best.pt"))
                log += "  * saved"

            if epoch % 5 == 0 or epoch == 1:
                print(f"  [sanity check epoch {epoch}]")
                sanity_check(model, val_loader, device, scalers, args.pred_mode)
        else:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "pos_mean": scalers.pos.mean, "pos_std": scalers.pos.std,
                "logdt_mean": scalers.logdt.mean, "logdt_std": scalers.logdt.std,
                "args": vars(args),
            }, os.path.join(args.out_dir, "last.pt"))

        print(log)

    print(f"\n  Best val MSE : {best_val_mse:.6f}")
    print(f"  Checkpoint   : {args.out_dir}/best.pt")
    print("=" * 65)


if __name__ == "__main__":
    main()