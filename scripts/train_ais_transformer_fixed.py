#!/usr/bin/env python3
"""
FIXED VERSION: AIS Transformer Regression
All critical bugs identified in debug report have been fixed.

Key fixes:
1. Added data validation and outlier removal
2. Fixed positional encoding flag implementation
3. Consistent dtype usage (float32 throughout)
4. Optional causal mask (default: bidirectional)
5. Proper sorting by time
6. Added velocity features option
7. Better scaler stability
8. Extensive logging and validation
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


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


@dataclass
class Scaler1D:
    """Scaler for 1D values (e.g. log(dt))."""
    mean: float
    std: float

    @staticmethod
    def fit(x: np.ndarray) -> "Scaler1D":
        mean = float(x.mean())
        std = float(x.std())
        if std < 1e-6:
            std = 1.0
        return Scaler1D(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class Scaler2D:
    """Scaler for 2D coordinates with improved stability."""
    mean: np.ndarray  # (2,)
    std: np.ndarray   # (2,)

    @staticmethod
    def fit(xy: np.ndarray) -> "Scaler2D":
        """Fit scaler on data.
        
        Args:
            xy: array of shape (N, 2) containing [lon, lat] coordinates
        """
        mean = xy.mean(axis=0).astype(np.float32)
        std = xy.std(axis=0).astype(np.float32)
        # More conservative threshold for stability
        std = np.where(std < 1e-6, 1.0, std)
        return Scaler2D(mean=mean, std=std)

    def transform(self, xy: np.ndarray) -> np.ndarray:
        """Transform coordinates to normalized space."""
        return (xy - self.mean) / self.std

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        """Transform normalized coordinates back to geographic space."""
        return xy * self.std + self.mean


def validate_coordinates(lon: np.ndarray, lat: np.ndarray, 
                         max_jump_degrees: float = 1.0) -> Tuple[bool, str]:
    """Validate coordinate arrays for a single track.
    
    Returns:
        (is_valid, reason) - True if valid, False otherwise with reason
    """
    # Check bounds
    if np.any(np.abs(lat) > 90):
        return False, "latitude out of bounds"
    if np.any(np.abs(lon) > 180):
        return False, "longitude out of bounds"
    
    # Check for NaN
    if np.any(np.isnan(lon)) or np.any(np.isnan(lat)):
        return False, "contains NaN"
    
    # Check for unrealistic jumps (>111km per step)
    if len(lon) > 1:
        dlon = np.diff(lon)
        dlat = np.diff(lat)
        jumps = np.sqrt(dlon**2 + dlat**2)
        if np.any(jumps > max_jump_degrees):
            max_jump = np.max(jumps)
            return False, f"unrealistic jump: {max_jump:.2f}° (~{max_jump*111:.0f}km)"
    
    return True, "valid"


def load_tracks(
    csv_path: str,
    id_col: str = "MMSI",
    time_col: str = "BaseDateTime",
    lat_col: str = "LAT",
    lon_col: str = "LON",
    max_jump_degrees: float = 1.0,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, "pd.Timestamp"]]:
    """Load vessel tracks from CSV with validation.
    
    Args:
        csv_path: Path to CSV file
        id_col: Column name for vessel ID
        time_col: Column name for timestamp
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        max_jump_degrees: Maximum allowed jump between consecutive points (degrees)
        verbose: Print loading statistics
    
    Returns:
        Dictionary mapping vessel_id -> trajectory array of shape (T, 3) with [lon, lat, dt_seconds].
        dt_seconds[0] is always 0 (no elapsed time before the first point).
    """
    df = pd.read_csv(csv_path)
    
    # Basic validation
    required_cols = [id_col, time_col, lat_col, lon_col]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows with NaN in required columns
    initial_rows = len(df)
    df = df.dropna(subset=required_cols)
    dropped_nan = initial_rows - len(df)
    
    # Filter invalid coordinates
    df = df[(df[lat_col].abs() <= 90) & (df[lon_col].abs() <= 180)]
    dropped_bounds = initial_rows - dropped_nan - len(df)
    
    # Parse timestamps and sort
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])  # Drop unparseable times
    df = df.sort_values([id_col, time_col])
    
    tracks: Dict[str, np.ndarray] = {}
    tracks_first_ts: Dict[str, pd.Timestamp] = {}
    validation_stats = {
        'total_vessels': 0,
        'valid_vessels': 0,
        'too_short': 0,
        'invalid_coords': 0,
        'unrealistic_jumps': 0,
    }
    
    for v_id, g in df.groupby(id_col):
        validation_stats['total_vessels'] += 1
        
        # Extract coordinates
        lon = g[lon_col].astype(float).to_numpy()
        lat = g[lat_col].astype(float).to_numpy()
        
        # Check minimum length
        if len(lon) < 5:
            validation_stats['too_short'] += 1
            continue
        
        # Validate coordinates
        is_valid, reason = validate_coordinates(lon, lat, max_jump_degrees)
        if not is_valid:
            if 'jump' in reason:
                validation_stats['unrealistic_jumps'] += 1
            else:
                validation_stats['invalid_coords'] += 1
            if verbose and validation_stats['total_vessels'] <= 10:
                print(f"  Skipping vessel {v_id}: {reason}")
            continue
        
        # Compute dt_seconds (time elapsed since previous point; 0 for the first)
        ts = g[time_col].to_numpy()
        dt = np.concatenate([[0.0], np.diff(ts.astype("datetime64[s]")).astype(float)])

        # Store as [lon, lat, dt_seconds] in float32
        pts = np.stack([lon, lat, dt], axis=1).astype(np.float32)
        tracks[str(v_id)] = pts
        # Record first timestamp for temporal splitting
        tracks_first_ts[str(v_id)] = g[time_col].iloc[0]
        validation_stats['valid_vessels'] += 1
    
    if verbose:
        print(f"\nData Loading Summary:")
        print(f"  Total rows in CSV: {initial_rows}")
        print(f"  Dropped (NaN): {dropped_nan}")
        print(f"  Dropped (out of bounds): {dropped_bounds}")
        print(f"  Total vessels: {validation_stats['total_vessels']}")
        print(f"  Valid vessels: {validation_stats['valid_vessels']}")
        print(f"  Skipped (too short <5 pts): {validation_stats['too_short']}")
        print(f"  Skipped (invalid coords): {validation_stats['invalid_coords']}")
        print(f"  Skipped (unrealistic jumps): {validation_stats['unrealistic_jumps']}")
        
        if validation_stats['valid_vessels'] > 0:
            lengths = [len(t) for t in tracks.values()]
            print(f"  Track lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    return tracks, tracks_first_ts


def train_val_split(tracks: Dict[str, np.ndarray], val_frac: float, seed: int):
    """Split tracks into train and validation sets."""
    ids = list(tracks.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    if len(ids) <= 1 or val_frac <= 0:
        return tracks, {}
    n_val = int(len(ids) * val_frac)
    val_ids = set(ids[:n_val])
    train = {k: v for k, v in tracks.items() if k not in val_ids}
    val = {k: v for k, v in tracks.items() if k in val_ids}
    return train, val


def temporal_train_val_split(
    tracks: Dict[str, np.ndarray],
    tracks_first_ts: Dict[str, "pd.Timestamp"],
    val_date: str,
):
    """Split tracks by date: vessels whose first point is before val_date go to train,
    vessels whose first point is on or after val_date go to val.

    This tests generalization to unseen future dates rather than unseen vessels
    from the same time period.
    """
    cutoff = pd.Timestamp(val_date)
    train = {k: v for k, v in tracks.items() if tracks_first_ts[k] < cutoff}
    val   = {k: v for k, v in tracks.items() if tracks_first_ts[k] >= cutoff}
    return train, val


class TrajectoryDataset(Dataset):
    """Sliding windows from vessel trajectories.

    Each sample contains:
        source_sequence: (seq_len, feature_dim) - input features [lon, lat, log_dt, (vel_lon, vel_lat)?]
        target: (seq_len, 2) - target positions [lon, lat]
    """

    def __init__(
        self,
        tracks: Dict[str, np.ndarray],
        scaler: Scaler2D,
        scaler_dt: Scaler1D,
        seq_len: int = 20,
        stride: int = 1,
        max_windows_per_track: Optional[int] = None,
        use_velocity: bool = False,
    ):
        self.tracks = tracks
        self.scaler = scaler
        self.scaler_dt = scaler_dt
        self.seq_len = seq_len
        self.stride = stride
        self.use_velocity = use_velocity
        self.index: List[Tuple[str, int]] = []

        for v_id, pts in tracks.items():
            T = pts.shape[0]
            if T < seq_len + 1:
                continue
            starts = list(range(0, T - (seq_len + 1) + 1, stride))
            if max_windows_per_track is not None and len(starts) > max_windows_per_track:
                starts = random.sample(starts, max_windows_per_track)
                starts.sort()
            for s in starts:
                self.index.append((v_id, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        v_id, s = self.index[index]
        pts = self.tracks[v_id][s : s + self.seq_len + 1]  # (seq_len+1, 3): [lon, lat, dt]

        # Input: points 0 to seq_len-1; Target: points 1 to seq_len
        x_raw = pts[:-1]  # (seq_len, 3)
        y = pts[1:, :2]   # (seq_len, 2) - target [lon, lat] only

        # Normalize position and time
        x_pos_norm = self.scaler.transform(x_raw[:, :2])          # (seq_len, 2)
        log_dt = np.log1p(np.clip(x_raw[:, 2], 0, None))          # (seq_len,)
        log_dt_norm = self.scaler_dt.transform(log_dt)[:, None]    # (seq_len, 1)
        y_norm = self.scaler.transform(y).astype(np.float32)

        if self.use_velocity:
            vel = np.diff(x_raw[:, :2], axis=0, prepend=x_raw[0:1, :2])  # (seq_len, 2)
            vel_norm = (vel / self.scaler.std).astype(np.float32)
            x = np.concatenate([x_pos_norm, log_dt_norm, vel_norm], axis=1)  # (seq_len, 5)
        else:
            x = np.concatenate([x_pos_norm, log_dt_norm], axis=1)  # (seq_len, 3)

        return {
            "source_sequence": torch.tensor(x),
            "target": torch.tensor(y_norm)
        }


def MyCollator(batch):
    """Collate batch of sequences."""
    x = torch.stack([sample["source_sequence"] for sample in batch], dim=0)
    y = torch.stack([sample["target"] for sample in batch], dim=0)
    # Transpose to (seq_len, batch_size, feature_dim)
    return x.transpose(0, 1), y.transpose(0, 1)


def get_loader(
    tracks: Dict[str, np.ndarray],
    scaler: Scaler2D,
    scaler_dt: Scaler1D,
    seq_len: int,
    batch_size: int,
    stride: int = 1,
    max_windows_per_track: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    use_velocity: bool = False,
):
    """Create DataLoader for trajectory prediction."""
    dataset = TrajectoryDataset(
        tracks=tracks,
        scaler=scaler,
        scaler_dt=scaler_dt,
        seq_len=seq_len,
        stride=stride,
        max_windows_per_track=max_windows_per_track,
        use_velocity=use_velocity,
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=MyCollator,
        drop_last=drop_last,
    )
    return data_loader


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, nhid, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, nhid)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, nhid, 2).float() * (-math.log(10000.0) / nhid)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer encoder for sequence modeling."""
    
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        nhid: int,
        nlayers: int,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, nhid)
        
        # Positional encoding (optional)
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(nhid, dropout)
        else:
            self.pos_encoder = None

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=nhid,
            nhead=nhead,
            dim_feedforward=4*nhid,  # Standard is 4x hidden dim
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)

        self.nhid = nhid
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive modeling."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.encoder.bias is not None:
            self.encoder.bias.data.zero_()

    def forward(self, src, src_mask=None):
        """Forward pass.
        
        Args:
            src: (seq_len, batch_size, input_dim)
            src_mask: Optional attention mask
        
        Returns:
            output: (seq_len, batch_size, nhid)
        """
        src = self.encoder(src) * math.sqrt(self.nhid)
        if self.pos_encoder is not None:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output


class RegressionHead(nn.Module):
    """Linear decoder for position prediction."""
    
    def __init__(self, nhid: int, out_dim: int = 2):
        super(RegressionHead, self).__init__()
        self.decoder = nn.Linear(nhid, out_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        return self.decoder(src)


class Model(nn.Module):
    """Complete model: Transformer + Regression head."""
    
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        nhid: int,
        nlayers: int,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super(Model, self).__init__()
        self.base = TransformerModel(
            input_dim=input_dim,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )
        self.regressor = RegressionHead(nhid, out_dim=2)

    def forward(self, src, src_mask=None):
        """Forward pass.
        
        Args:
            src: (seq_len, batch_size, input_dim)
            src_mask: Optional attention mask
        
        Returns:
            output: (seq_len, batch_size, 2) - predicted [lon, lat]
        """
        x = self.base(src, src_mask)
        output = self.regressor(x)
        return output


def haversine_meters(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Compute Haversine distance in meters.
    
    Args:
        lat1, lon1, lat2, lon2: Arrays in degrees
        
    Returns:
        Distance in meters
    """
    r = 6371000.0  # Earth's radius in meters
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


@torch.no_grad()
def eval_regression(model: Model, loader: DataLoader, device: torch.device, 
                    scaler: Scaler2D, use_causal_mask: bool = False):
    """Evaluate model on validation set.
    
    Returns:
        Dictionary with metrics: mse, ade_m, fde_m
    """
    model.eval()
    mse_fn = nn.MSELoss(reduction="mean")

    total_mse = 0.0
    n_batches = 0
    ade_sum = 0.0
    fde_sum = 0.0

    for src, target in loader:
        src = src.to(device)
        target = target.to(device)
        
        src_mask = None
        if use_causal_mask:
            src_mask = model.base.generate_square_subsequent_mask(src.size(0)).to(device)
        
        out = model(src, src_mask)

        total_mse += float(mse_fn(out, target).item())
        n_batches += 1

        # Convert to geographic coordinates for distance metrics
        out_np = out.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()

        # Inverse transform: normalized -> [lon, lat]
        out_ll = scaler.inverse(out_np.reshape(-1, 2)).reshape(out_np.shape)
        tgt_ll = scaler.inverse(tgt_np.reshape(-1, 2)).reshape(tgt_np.shape)

        # Compute Haversine distance
        dist = haversine_meters(
            lat1=tgt_ll[:, :, 1], lon1=tgt_ll[:, :, 0],
            lat2=out_ll[:, :, 1], lon2=out_ll[:, :, 0],
        )
        ade_sum += float(dist.mean())

        # Final displacement error (last timestep only)
        dist_last = haversine_meters(
            lat1=tgt_ll[-1, :, 1], lon1=tgt_ll[-1, :, 0],
            lat2=out_ll[-1, :, 1], lon2=out_ll[-1, :, 0],
        )
        fde_sum += float(dist_last.mean())

    if n_batches == 0:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}

    return {
        "mse": total_mse / n_batches,
        "ade_m": ade_sum / n_batches,
        "fde_m": fde_sum / n_batches,
    }


@torch.no_grad()
def eval_constant_velocity_baseline(loader: DataLoader, device: torch.device, scaler: Scaler2D):
    """Evaluate constant velocity baseline.
    
    Predicts next position as: pos[t+1] = pos[t] + (pos[t] - pos[t-1])
    """
    mse_fn = nn.MSELoss(reduction="mean")
    total_mse = 0.0
    n_batches = 0
    ade_sum = 0.0
    fde_sum = 0.0

    for src, target in loader:
        src = src.to(device)
        target = target.to(device)

        # Extract position features only (first 2 dimensions)
        src_pos = src[..., :2]
        
        # Constant velocity prediction
        pred = torch.zeros_like(target)
        pred[0] = src_pos[0]
        pred[1:] = src_pos[1:] + (src_pos[1:] - src_pos[:-1])

        total_mse += float(mse_fn(pred, target).item())
        n_batches += 1

        pred_np = pred.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()

        pred_ll = scaler.inverse(pred_np.reshape(-1, 2)).reshape(pred_np.shape)
        tgt_ll = scaler.inverse(tgt_np.reshape(-1, 2)).reshape(tgt_np.shape)

        dist = haversine_meters(
            lat1=tgt_ll[:, :, 1], lon1=tgt_ll[:, :, 0],
            lat2=pred_ll[:, :, 1], lon2=pred_ll[:, :, 0],
        )
        ade_sum += float(dist.mean())

        dist_last = haversine_meters(
            lat1=tgt_ll[-1, :, 1], lon1=tgt_ll[-1, :, 0],
            lat2=pred_ll[-1, :, 1], lon2=pred_ll[-1, :, 0],
        )
        fde_sum += float(dist_last.mean())

    if n_batches == 0:
        return {"mse": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}

    return {
        "mse": total_mse / n_batches,
        "ade_m": ade_sum / n_batches,
        "fde_m": fde_sum / n_batches,
    }


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Linear warmup then cosine annealing LR schedule."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train(
    model: Model,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    scheduler: Optional[LambdaLR] = None,
    log_interval: int = 20,
    current_epoch: int = 1,
    grad_clip: float = 0.5,
    use_causal_mask: bool = True,
):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    mse_fn = nn.MSELoss(reduction="mean")

    for idx, data in enumerate(loader):
        optimizer.zero_grad()

        src, target = data  # (seq_len, batch_size, feature_dim), (seq_len, batch_size, 2)
        src = src.to(device)
        target = target.to(device)

        src_mask = None
        if use_causal_mask:
            src_mask = model.base.generate_square_subsequent_mask(src.size(0)).to(device)

        output = model(src, src_mask)  # (seq_len, batch_size, 2)

        loss = mse_fn(output, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())

        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
            print(
                f"| epoch {current_epoch:3d} | {idx:5d}/{len(loader):5d} steps"
                f" | mse {cur_loss:8.6f} | grad {grad_norm:.3f} | lr {lr:.2e}"
            )
            total_loss = 0.0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="FIXED: Transformer for AIS trajectory prediction"
    )
    
    # Data args
    parser.add_argument("--csv", required=True, help="Path to AIS CSV file")
    parser.add_argument("--out_dir", default="runs/ais_transformer_fixed")
    parser.add_argument("--id_col", default="MMSI")
    parser.add_argument("--time_col", default="BaseDateTime")
    parser.add_argument("--lat_col", default="LAT")
    parser.add_argument("--lon_col", default="LON")
    parser.add_argument("--max_jump_degrees", type=float, default=1.0,
                       help="Max allowed jump between points (degrees, ~111km)")
    
    # Sequence args
    parser.add_argument("--seq_len", type=int, default=20,
                       help="Sequence length (context window for the transformer)")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_windows_per_track", type=int, default=None)
    
    # Features
    parser.add_argument("--use_velocity", action="store_true",
                       help="Add velocity features to input")
    
    # Split
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of vessels for validation (used when --val_date is not set)")
    parser.add_argument("--val_date", type=str, default=None,
                        help="If set (e.g. '2024-03-28'), vessels whose first point is on or "
                             "after this date go to val; all others go to train. "
                             "Use this for multi-day data (recommended over --val_frac).")
    parser.add_argument("--seed", type=int, default=0)
    
    # Model args
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_positional_encoding", action="store_false", dest="use_positional_encoding",
                       help="Disable sinusoidal positional encoding (enabled by default)")
    parser.set_defaults(use_positional_encoding=True)
    parser.add_argument("--no_causal_mask", action="store_false", dest="use_causal_mask",
                       help="Disable causal mask / use bidirectional attention (causal is default)")
    parser.set_defaults(use_causal_mask=True)
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.5)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}\n")

    # Load and validate data
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    tracks, tracks_first_ts = load_tracks(
        args.csv,
        id_col=args.id_col,
        time_col=args.time_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        max_jump_degrees=args.max_jump_degrees,
        verbose=True,
    )

    if len(tracks) == 0:
        raise RuntimeError("No valid tracks loaded. Check your data!")

    # Train/val split
    if args.val_date:
        train_tracks, val_tracks = temporal_train_val_split(
            tracks, tracks_first_ts, args.val_date
        )
        print(f"\nTemporal split on {args.val_date}: "
              f"{len(train_tracks)} train vessels, {len(val_tracks)} val vessels")
    else:
        train_tracks, val_tracks = train_val_split(tracks, args.val_frac, args.seed)
        print(f"\nRandom split: {len(train_tracks)} train vessels, {len(val_tracks)} val vessels")

    if len(train_tracks) == 0:
        raise RuntimeError("No training tracks after split!")

    # Fit scalers on training data only
    all_train_pts = np.concatenate(list(train_tracks.values()), axis=0)
    scaler = Scaler2D.fit(all_train_pts[:, :2])   # fit on [lon, lat]
    log_dt_all = np.log1p(np.clip(all_train_pts[:, 2], 0, None))
    scaler_dt = Scaler1D.fit(log_dt_all)
    print(f"\nScaler fitted:")
    print(f"  Mean (lon, lat):   [{scaler.mean[0]:.4f}, {scaler.mean[1]:.4f}]")
    print(f"  Std  (lon, lat):   [{scaler.std[0]:.4f}, {scaler.std[1]:.4f}]")
    print(f"  Mean log(dt):      {scaler_dt.mean:.4f}")
    print(f"  Std  log(dt):      {scaler_dt.std:.4f}")

    # Determine input dimension: [lon, lat, log_dt] = 3, + [vel_lon, vel_lat] = 5
    input_dim = 5 if args.use_velocity else 3
    print(f"\nModel input dimension: {input_dim} {'(pos + log_dt + vel)' if args.use_velocity else '(pos + log_dt)'}")

    # Create data loaders
    train_loader = get_loader(
        tracks=train_tracks,
        scaler=scaler,
        scaler_dt=scaler_dt,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        stride=args.stride,
        max_windows_per_track=args.max_windows_per_track,
        shuffle=True,
        drop_last=True,
        use_velocity=args.use_velocity,
    )
    print(f"Training batches: {len(train_loader)}")

    val_loader = None
    if len(val_tracks) > 0:
        val_loader = get_loader(
            tracks=val_tracks,
            scaler=scaler,
            scaler_dt=scaler_dt,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            stride=args.stride,
            max_windows_per_track=args.max_windows_per_track,
            shuffle=False,
            drop_last=False,
            use_velocity=args.use_velocity,
        )
        print(f"Validation batches: {len(val_loader)}")

    # Create model
    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE")
    print("="*60)
    model = Model(
        input_dim=input_dim,
        nhead=args.nhead,
        nhid=args.nhid,
        nlayers=args.nlayers,
        dropout=args.dropout,
        use_positional_encoding=args.use_positional_encoding,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Positional encoding: {args.use_positional_encoding}")
    print(f"Causal mask: {args.use_causal_mask}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(int(0.05 * total_steps), 500)  # 5% warmup, capped at 500 steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"LR schedule: cosine with {warmup_steps} warmup steps ({total_steps} total)")

    os.makedirs(args.out_dir, exist_ok=True)

    # Evaluate baseline before training
    baseline_metrics = None
    if val_loader is not None:
        print(f"\n{'='*60}")
        print("BASELINE (Constant Velocity)")
        print("="*60)
        baseline_metrics = eval_constant_velocity_baseline(val_loader, device, scaler)
        print(f"Val MSE: {baseline_metrics['mse']:.6f} | ADE: {baseline_metrics['ade_m']:.1f}m | FDE: {baseline_metrics['fde_m']:.1f}m")
        print("\nIf your model doesn't beat this baseline, something is wrong!\n")

    best_val = float("inf")
    best_metrics = None

    # Training loop
    print(f"{'='*60}")
    print("TRAINING")
    print("="*60)
    for epoch in range(1, args.epochs + 1):
        train(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            scheduler=scheduler,
            log_interval=20,
            current_epoch=epoch,
            grad_clip=args.grad_clip,
            use_causal_mask=args.use_causal_mask,
        )

        if val_loader is not None:
            metrics = eval_regression(
                model, val_loader, device, scaler,
                use_causal_mask=args.use_causal_mask,
            )
            print(
                f"Eval | epoch {epoch:3d} | val mse {metrics['mse']:.6f} | "
                f"ADE {metrics['ade_m']:.1f}m | FDE {metrics['fde_m']:.1f}m"
            )

            if metrics["mse"] < best_val:
                best_val = metrics["mse"]
                best_metrics = metrics
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_mean': scaler.mean,
                    'scaler_std': scaler.std,
                    'scaler_dt_mean': scaler_dt.mean,
                    'scaler_dt_std': scaler_dt.std,
                    'args': vars(args),
                    'metrics': metrics,
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "best_model.pt"))
                print(f"  → Saved checkpoint (best val MSE: {best_val:.6f})")
        else:
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation MSE: {best_val:.6f}")
    print(f"Models saved to: {args.out_dir}")

    # Final comparison vs baseline
    if baseline_metrics is not None and best_metrics is not None:
        print(f"\n{'='*60}")
        print("FINAL: MODEL vs BASELINE")
        print("="*60)
        ade_delta = best_metrics['ade_m'] - baseline_metrics['ade_m']
        fde_delta = best_metrics['fde_m'] - baseline_metrics['fde_m']
        ade_pct = 100 * ade_delta / (baseline_metrics['ade_m'] + 1e-9)
        fde_pct = 100 * fde_delta / (baseline_metrics['fde_m'] + 1e-9)
        result = "BEATS" if ade_delta < 0 else "LOSES TO"
        print(f"              ADE (m)     FDE (m)")
        print(f"  Baseline:  {baseline_metrics['ade_m']:8.1f}   {baseline_metrics['fde_m']:8.1f}")
        print(f"  Model:     {best_metrics['ade_m']:8.1f}   {best_metrics['fde_m']:8.1f}")
        print(f"  Delta:     {ade_delta:+8.1f}   {fde_delta:+8.1f}  ({ade_pct:+.1f}% / {fde_pct:+.1f}%)")
        print(f"\n  Model {result} the constant-velocity baseline.")


if __name__ == "__main__":
    main()
