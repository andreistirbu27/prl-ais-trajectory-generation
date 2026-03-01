#!/usr/bin/env python3
"""
train_ais_transformer.py — AIS Vessel Trajectory Prediction

Transformer encoder that predicts the next position for each timestep
in a sliding window sequence (seq2seq, shifted by 1).

Input features per timestep  : [lon, lat, dt]         (3-dim, default)
                               + [dlon, dlat]          (5-dim with --use_velocity)
Target per timestep          : [lon, lat]              (2-dim, shifted +1)

Key design decisions:
  - Bidirectional attention (no causal mask): consistent with how the model
    is used at inference time (whole window in, whole shifted window out).
  - dt (time-delta) as an explicit input feature: your data has variable gaps
    (mean 102s, p95 181s) so the model needs to know how much time passed.
  - Cosine LR schedule with linear warmup.
  - Full checkpoint saves args + scaler + metrics so the model is self-contained.

Usage:
    python train_ais_transformer.py --csv data/processed/ais.csv
    python train_ais_transformer.py --csv data/processed/ais.csv \\
        --seq_len 20 --nhid 256 --nlayers 4 --epochs 30 --use_velocity
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
# Scaler
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scaler:
    """Per-feature mean/std normalisation. Fitted on training data only."""
    mean: np.ndarray  # (feature_dim,)
    std: np.ndarray   # (feature_dim,)

    @staticmethod
    def fit(x: np.ndarray) -> "Scaler":
        """
        Args:
            x: (N, feature_dim)
        """
        mean = x.mean(axis=0).astype(np.float32)
        std  = x.std(axis=0).astype(np.float32)
        std  = np.where(std < 1e-6, 1.0, std)   # avoid div-by-zero for constant features
        return Scaler(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_pos(self, xy: np.ndarray) -> np.ndarray:
        """Inverse-transform position features only (first 2 dims: lon, lat)."""
        return (xy * self.std[:2] + self.mean[:2]).astype(np.float32)


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
    """
    Load processed AIS CSV into per-vessel trajectory arrays.

    Returns:
        dict: mmsi -> float32 array of shape (T, 3) = [lon, lat, dt_seconds]
              dt for the first point of each track is 0.
    """
    df = pd.read_csv(csv_path)

    required = [id_col, time_col, lat_col, lon_col]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=required)
    df = df.sort_values([id_col, time_col])

    tracks: Dict[str, np.ndarray] = {}
    for v_id, g in df.groupby(id_col):
        lon = g[lon_col].to_numpy(dtype=np.float32)
        lat = g[lat_col].to_numpy(dtype=np.float32)
        ts  = g[time_col].values.astype("datetime64[s]").astype(np.float64)
        dt  = np.concatenate([[0.0], np.diff(ts)]).astype(np.float32)  # seconds; 0 for first point
        pts = np.stack([lon, lat, dt], axis=1)  # (T, 3)
        if pts.shape[0] >= 5:
            tracks[str(v_id)] = pts

    print(f"Loaded {len(tracks):,} vessels from {csv_path}")
    lengths = [v.shape[0] for v in tracks.values()]
    print(f"  Track length — min: {min(lengths)}  median: {np.median(lengths):.0f}  max: {max(lengths)}")
    return tracks


# ─────────────────────────────────────────────────────────────────────────────
# Train / val split
# ─────────────────────────────────────────────────────────────────────────────

def train_val_split(
    tracks: Dict[str, np.ndarray],
    val_frac: float,
    seed: int,
) -> Tuple[Dict, Dict]:
    ids = list(tracks.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    if len(ids) <= 1 or val_frac <= 0:
        return tracks, {}
    n_val  = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    train  = {k: v for k, v in tracks.items() if k not in val_ids}
    val    = {k: v for k, v in tracks.items() if k in val_ids}
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Sliding-window dataset over vessel trajectories.

    Each sample:
        x : (seq_len, input_dim)   — normalised input features
        y : (seq_len, 2)           — normalised [lon, lat] shifted by 1 step
    """

    def __init__(
        self,
        tracks: Dict[str, np.ndarray],
        scaler: Scaler,
        seq_len: int,
        stride: int = 1,
        max_windows_per_track: Optional[int] = None,
        use_velocity: bool = False,
    ):
        self.tracks    = tracks
        self.scaler    = scaler
        self.seq_len   = seq_len
        self.use_velocity = use_velocity
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

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        v_id, s = self.index[idx]
        pts = self.tracks[v_id][s : s + self.seq_len + 1]  # (seq_len+1, 3)

        # Input window: positions 0..seq_len-1
        # Target window: positions 1..seq_len  (lon, lat only)
        src_pts = pts[:-1]   # (seq_len, 3)  [lon, lat, dt]
        tgt_pts = pts[1:, :2]  # (seq_len, 2)  [lon, lat]

        # Build input features
        src_norm = self.scaler.transform(src_pts)   # (seq_len, 3) normalised

        if self.use_velocity:
            # Velocity = positional diff; use scaler std for consistent units
            # Shape: (seq_len, 2)
            dpos = np.diff(src_pts[:, :2], axis=0, prepend=src_pts[0:1, :2])
            vel_norm = (dpos / self.scaler.std[:2]).astype(np.float32)
            x = np.concatenate([src_norm, vel_norm], axis=1)   # (seq_len, 5)
        else:
            x = src_norm   # (seq_len, 3)

        # Target: normalise lon/lat only (first 2 dims of scaler)
        y = self.scaler.transform(
            np.concatenate([tgt_pts, np.zeros((self.seq_len, 1), dtype=np.float32)], axis=1)
        )[:, :2]   # (seq_len, 2)

        return torch.tensor(x), torch.tensor(y)


def collate_fn(batch):
    """Stack and transpose to (seq_len, batch, dim) for nn.Transformer."""
    xs, ys = zip(*batch)
    x = torch.stack(xs).transpose(0, 1)   # (seq_len, B, input_dim)
    y = torch.stack(ys).transpose(0, 1)   # (seq_len, B, 2)
    return x, y


def get_loader(
    tracks: Dict[str, np.ndarray],
    scaler: Scaler,
    seq_len: int,
    batch_size: int,
    stride: int = 1,
    max_windows_per_track: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    use_velocity: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TrajectoryDataset(
        tracks=tracks,
        scaler=scaler,
        seq_len=seq_len,
        stride=stride,
        max_windows_per_track=max_windows_per_track,
        use_velocity=use_velocity,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))   # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AISTransformer(nn.Module):
    """
    Transformer encoder for AIS trajectory prediction.

    Architecture:
        Linear projection → PositionalEncoding → TransformerEncoder → Linear head

    No causal mask: the model sees the whole input window and predicts the
    whole shifted-by-1 window in one forward pass (bidirectional attention).
    This is consistent with how the model is evaluated at inference time.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int    = 128,
        nhead: int      = 8,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float  = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        dim_feedforward = dim_feedforward or 4 * d_model   # standard transformer ratio

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 2)   # predict [lon, lat]

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.input_proj.weight, -0.1, 0.1)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.uniform_(self.output_proj.weight, -0.1, 0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, input_dim)
        Returns:
            (seq_len, batch, 2)  — predicted [lon, lat] in normalised space
        """
        x = self.input_proj(x) * math.sqrt(x.size(-1))
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.output_proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def haversine_meters(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine distance in metres between two sets of (lat, lon) points."""
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: AISTransformer,
    loader: DataLoader,
    device: torch.device,
    scaler: Scaler,
) -> dict:
    """
    Returns:
        mse   : mean squared error in normalised space
        ade_m : Average Displacement Error in metres (mean over all timesteps)
        fde_m : Final Displacement Error in metres (last timestep only)
    """
    model.eval()
    mse_fn = nn.MSELoss()
    total_mse, ade_sum, fde_sum, n = 0.0, 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)                        # (seq_len, B, 2)

        total_mse += mse_fn(pred, y).item()
        n += 1

        # Convert back to geographic coords for interpretable metrics
        pred_geo = scaler.inverse_pos(pred.cpu().numpy().reshape(-1, 2)).reshape(pred.shape)
        y_geo    = scaler.inverse_pos(y.cpu().numpy().reshape(-1, 2)).reshape(y.shape)

        # pred_geo / y_geo shape: (seq_len, B, 2) = [..., lon/lat]
        dist = haversine_meters(y_geo[:, :, 1], y_geo[:, :, 0],
                                pred_geo[:, :, 1], pred_geo[:, :, 0])
        ade_sum += dist.mean()
        fde_sum += dist[-1].mean()

    if n == 0:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}
    return {"mse": total_mse / n, "ade_m": ade_sum / n, "fde_m": fde_sum / n}


@torch.no_grad()
def evaluate_constant_velocity_baseline(
    loader: DataLoader,
    device: torch.device,
    scaler: Scaler,
) -> dict:
    """
    Naive baseline: predict next pos = current pos + last observed delta.
    Your model must beat this to be useful.
    """
    mse_fn = nn.MSELoss()
    total_mse, ade_sum, fde_sum, n = 0.0, 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # x: (seq_len, B, input_dim) — first 2 dims are normalised lon/lat
        pos = x[:, :, :2]                          # (seq_len, B, 2)
        vel = torch.diff(pos, dim=0, prepend=pos[:1])
        pred = pos + vel                            # constant velocity extrapolation

        total_mse += mse_fn(pred, y).item()
        n += 1

        pred_geo = scaler.inverse_pos(pred.cpu().numpy().reshape(-1, 2)).reshape(pred.shape)
        y_geo    = scaler.inverse_pos(y.cpu().numpy().reshape(-1, 2)).reshape(y.shape)

        dist = haversine_meters(y_geo[:, :, 1], y_geo[:, :, 0],
                                pred_geo[:, :, 1], pred_geo[:, :, 0])
        ade_sum += dist.mean()
        fde_sum += dist[-1].mean()

    if n == 0:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}
    return {"mse": total_mse / n, "ade_m": ade_sum / n, "fde_m": fde_sum / n}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: AISTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    log_every: int = 50,
) -> float:
    model.train()
    mse_fn = nn.MSELoss()
    running, total, n = 0.0, 0.0, 0

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = mse_fn(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        running += loss.item()
        total   += loss.item()
        n       += 1

        if (i + 1) % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch:3d} | step {i+1:5d}/{len(loader)} "
                  f"| mse {running/log_every:.6f} | lr {lr:.2e}")
            running = 0.0

    return total / max(n, 1)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay to 1e-5."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-5, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AIS vessel trajectory prediction — Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--csv",       required=True)
    parser.add_argument("--out_dir",   default="runs/ais_transformer")
    parser.add_argument("--id_col",    default="MMSI")
    parser.add_argument("--time_col",  default="BaseDateTime")
    parser.add_argument("--lat_col",   default="LAT")
    parser.add_argument("--lon_col",   default="LON")

    # Sequence
    parser.add_argument("--seq_len",               type=int,   default=20)
    parser.add_argument("--stride",                type=int,   default=5,
                        help="Window stride. Larger = fewer (but more independent) windows.")
    parser.add_argument("--max_windows_per_track", type=int,   default=None)
    parser.add_argument("--use_velocity",          action="store_true",
                        help="Append [dlon, dlat] velocity features to input.")

    # Split
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed",     type=int,   default=42)

    # Model
    parser.add_argument("--d_model",         type=int,   default=128)
    parser.add_argument("--nhead",           type=int,   default=8)
    parser.add_argument("--num_layers",      type=int,   default=2)
    parser.add_argument("--dim_feedforward", type=int,   default=None,
                        help="FFN hidden dim. Defaults to 4 * d_model.")
    parser.add_argument("--dropout",         type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--warmup_frac",  type=float, default=0.05,
                        help="Fraction of total steps used for LR warmup.")
    parser.add_argument("--num_workers",  type=int,   default=0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  AIS TRANSFORMER — TRAINING")
    print("=" * 65)
    print(f"  Device : {device}")
    print(f"  Out dir: {args.out_dir}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    tracks = load_tracks(
        args.csv,
        id_col=args.id_col,
        time_col=args.time_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
    )
    if not tracks:
        raise RuntimeError("No valid tracks loaded.")

    train_tracks, val_tracks = train_val_split(tracks, args.val_frac, args.seed)
    print(f"  Train vessels: {len(train_tracks):,}  |  Val vessels: {len(val_tracks):,}\n")

    # ── Fit scaler on training data only ────────────────────────────────────
    all_train = np.concatenate(list(train_tracks.values()), axis=0)  # (N, 3)
    scaler = Scaler.fit(all_train)
    print(f"  Scaler (lon, lat, dt_sec):")
    print(f"    mean = {scaler.mean}")
    print(f"    std  = {scaler.std}\n")

    # ── Determine input dimension ────────────────────────────────────────────
    # Base: [lon, lat, dt] = 3 dims
    # With velocity: + [dlon, dlat] = 5 dims
    input_dim = 5 if args.use_velocity else 3
    print(f"  Input dim: {input_dim} {'(pos + dt + vel)' if args.use_velocity else '(pos + dt)'}")

    # ── Data loaders ─────────────────────────────────────────────────────────
    train_loader = get_loader(
        train_tracks, scaler,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        stride=args.stride,
        max_windows_per_track=args.max_windows_per_track,
        shuffle=True, drop_last=True,
        use_velocity=args.use_velocity,
        num_workers=args.num_workers,
    )
    val_loader = get_loader(
        val_tracks, scaler,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        stride=args.stride,
        max_windows_per_track=args.max_windows_per_track,
        shuffle=False, drop_last=False,
        use_velocity=args.use_velocity,
        num_workers=args.num_workers,
    ) if val_tracks else None

    print(f"  Train batches: {len(train_loader):,}")
    if val_loader:
        print(f"  Val batches:   {len(val_loader):,}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = AISTransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params: {n_params:,}")
    print(f"  d_model={args.d_model}  nhead={args.nhead}  "
          f"layers={args.num_layers}  ffn={args.dim_feedforward or 4*args.d_model}")

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler    = build_scheduler(optimizer, warmup_steps, total_steps)
    print(f"  Total steps: {total_steps:,}  |  Warmup steps: {warmup_steps:,}\n")

    # ── Constant velocity baseline ───────────────────────────────────────────
    if val_loader:
        print("─" * 65)
        baseline = evaluate_constant_velocity_baseline(val_loader, device, scaler)
        print(f"  Constant-velocity baseline:")
        print(f"    MSE {baseline['mse']:.6f}  |  "
              f"ADE {baseline['ade_m']:.1f} m  |  FDE {baseline['fde_m']:.1f} m")
        print(f"  Your model must beat this. If it doesn't, something is wrong.")
        print("─" * 65 + "\n")

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.grad_clip, epoch,
        )

        log = f"Epoch {epoch:3d}/{args.epochs} | train mse {train_mse:.6f}"

        if val_loader:
            metrics = evaluate(model, val_loader, device, scaler)
            log += (f" | val mse {metrics['mse']:.6f} | "
                    f"ADE {metrics['ade_m']:.1f} m | FDE {metrics['fde_m']:.1f} m")

            if metrics["mse"] < best_val_mse:
                best_val_mse = metrics["mse"]
                checkpoint = {
                    "epoch":       epoch,
                    "model":       model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scaler_mean": scaler.mean,
                    "scaler_std":  scaler.std,
                    "metrics":     metrics,
                    "args":        vars(args),
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "best.pt"))
                log += "  ✓ saved"
        else:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "scaler_mean": scaler.mean, "scaler_std": scaler.std,
                "args": vars(args),
            }, os.path.join(args.out_dir, "last.pt"))

        print(log)

    print(f"\n  Best val MSE : {best_val_mse:.6f}")
    print(f"  Checkpoint   : {args.out_dir}/best.pt")
    print("=" * 65)


if __name__ == "__main__":
    main()