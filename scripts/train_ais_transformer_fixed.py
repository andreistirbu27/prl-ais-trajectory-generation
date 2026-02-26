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
) -> Dict[str, np.ndarray]:
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
        Dictionary mapping vessel_id -> trajectory array of shape (T, 2) with [lon, lat]
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
        
        # Store as [lon, lat] in float32
        pts = np.stack([lon, lat], axis=1).astype(np.float32)
        tracks[str(v_id)] = pts
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
    
    return tracks


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


class TrajectoryDataset(Dataset):
    """Sliding windows from vessel trajectories.
    
    Each sample contains:
        source_sequence: (seq_len, feature_dim) - input features
        target: (seq_len, 2) - target positions [lon, lat]
    """

    def __init__(
        self,
        tracks: Dict[str, np.ndarray],
        scaler: Scaler2D,
        seq_len: int = 20,
        stride: int = 1,
        max_windows_per_track: Optional[int] = None,
        use_velocity: bool = False,
    ):
        self.tracks = tracks
        self.scaler = scaler
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
        pts = self.tracks[v_id][s : s + self.seq_len + 1]  # (seq_len+1, 2)
        
        # Input: points 0 to seq_len-1
        # Target: points 1 to seq_len
        x_pos = pts[:-1]  # (seq_len, 2) - positions
        y = pts[1:]       # (seq_len, 2) - target positions
        
        # Normalize positions
        x_pos_norm = self.scaler.transform(x_pos).astype(np.float32)
        y_norm = self.scaler.transform(y).astype(np.float32)
        
        if self.use_velocity:
            # Compute velocities (difference between consecutive positions)
            vel = np.diff(pts[:-1], axis=0, prepend=pts[0:1])  # (seq_len, 2)
            vel_norm = vel / self.scaler.std  # Normalize by std only
            
            # Concatenate: [pos_lon, pos_lat, vel_lon, vel_lat]
            x = np.concatenate([x_pos_norm, vel_norm.astype(np.float32)], axis=1)
        else:
            x = x_pos_norm
        
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


def train(
    model: Model,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    log_interval: int = 20,
    current_epoch: int = 1,
    grad_clip: float = 0.5,
    use_causal_mask: bool = False,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())

        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            print(
                f"| epoch {current_epoch:3d} | {idx:5d}/{len(loader):5d} steps | mse {cur_loss:8.6f}"
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
    parser.add_argument("--seq_len", type=int, default=10,
                       help="Sequence length (REDUCED from 20 to 10)")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_windows_per_track", type=int, default=None)
    
    # Features
    parser.add_argument("--use_velocity", action="store_true",
                       help="Add velocity features to input")
    
    # Split
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    
    # Model args
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_positional_encoding", action="store_true",
                       help="Use sinusoidal positional encoding")
    parser.add_argument("--use_causal_mask", action="store_true",
                       help="Use causal mask (default: bidirectional attention)")
    
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
    tracks = load_tracks(
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
    train_tracks, val_tracks = train_val_split(tracks, args.val_frac, args.seed)
    print(f"\nSplit: {len(train_tracks)} train vessels, {len(val_tracks)} val vessels")

    if len(train_tracks) == 0:
        raise RuntimeError("No training tracks after split!")

    # Fit scaler on training data
    all_train_pts = np.concatenate(list(train_tracks.values()), axis=0)
    scaler = Scaler2D.fit(all_train_pts)
    print(f"\nScaler fitted:")
    print(f"  Mean (lon, lat): [{scaler.mean[0]:.4f}, {scaler.mean[1]:.4f}]")
    print(f"  Std  (lon, lat): [{scaler.std[0]:.4f}, {scaler.std[1]:.4f}]")

    # Determine input dimension
    input_dim = 4 if args.use_velocity else 2
    print(f"\nModel input dimension: {input_dim} {'(pos + vel)' if args.use_velocity else '(pos only)'}")
    
    # Create data loaders
    train_loader = get_loader(
        tracks=train_tracks,
        scaler=scaler,
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
        weight_decay=args.weight_decay
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Evaluate baseline
    if val_loader is not None:
        print(f"\n{'='*60}")
        print("BASELINE (Constant Velocity)")
        print("="*60)
        base = eval_constant_velocity_baseline(val_loader, device, scaler)
        print(f"Val MSE: {base['mse']:.6f} | ADE: {base['ade_m']:.1f}m | FDE: {base['fde_m']:.1f}m")
        print("\nIf your model doesn't beat this baseline, something is wrong!\n")

    best_val = float("inf")

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
            log_interval=20,
            current_epoch=epoch,
            grad_clip=args.grad_clip,
            use_causal_mask=args.use_causal_mask,
        )

        if val_loader is not None:
            metrics = eval_regression(
                model, val_loader, device, scaler, 
                use_causal_mask=args.use_causal_mask
            )
            print(
                f"Eval | epoch {epoch:3d} | val mse {metrics['mse']:.6f} | "
                f"ADE {metrics['ade_m']:.1f}m | FDE {metrics['fde_m']:.1f}m"
            )

            if metrics["mse"] < best_val:
                best_val = metrics["mse"]
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_mean': scaler.mean,
                    'scaler_std': scaler.std,
                    'args': vars(args),
                    'metrics': metrics,
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "best_model.pt"))
                print(f"  → Saved checkpoint (best val MSE: {best_val:.6f})")
        else:
            # No validation set, save every epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation MSE: {best_val:.6f}")
    print(f"Models saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
