"""
AIS data loading, normalization, and dataset classes.

Tracks are stored as float32 arrays of shape (T, 3): [lon, lat, dt_seconds].
dt_seconds[0] is always 0 (no elapsed time before the first point).

Three scalers are maintained (all fit on training data only):
  pos   — [lon, lat] absolute position, spans the entire region
  logdt — log1p(dt_seconds), log-transformed to remove right-skew
  disp  — [dlon, dlat] per-step displacement, small variance centered near zero

Vessel type: each track also carries an AIS vessel type code (int, 0=unknown).
A vocab dict (raw_code → idx) maps raw codes to embedding indices.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ─── Scalers ─────────────────────────────────────────────────────────────────

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
    disp:  Scaler   # [dlon, dlat] per-step displacement

    @staticmethod
    def fit(tracks: Dict[str, np.ndarray]) -> "Scalers":
        all_pos, all_logdt, all_disp = [], [], []
        for pts in tracks.values():
            all_pos.append(pts[:, :2])
            all_logdt.append(np.log1p(pts[:, 2]).reshape(-1, 1))
            all_disp.append(np.diff(pts[:, :2], axis=0))  # (T-1, 2)
        return Scalers(
            pos   = Scaler.fit(np.concatenate(all_pos,   axis=0)),
            logdt = Scaler.fit(np.concatenate(all_logdt, axis=0)),
            disp  = Scaler.fit(np.concatenate(all_disp,  axis=0)),
        )


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_tracks(
    csv_path: str,
    id_col:          str = "MMSI",
    time_col:        str = "BaseDateTime",
    lat_col:         str = "LAT",
    lon_col:         str = "LON",
    vessel_type_col: str = "VesselType",
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Returns (tracks, vessel_types) where:
      tracks       : vessel_id -> float32 (T, 3) = [lon, lat, dt_seconds]
      vessel_types : vessel_id -> int AIS type code (0 = unknown/missing)
    """
    df = pd.read_csv(csv_path)
    missing = set([id_col, time_col, lat_col, lon_col]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[id_col, time_col, lat_col, lon_col])
    df = df.sort_values([id_col, time_col])

    tracks: Dict[str, np.ndarray] = {}
    vessel_types: Dict[str, int] = {}
    for v_id, g in df.groupby(id_col):
        lon = g[lon_col].to_numpy(dtype=np.float32)
        lat = g[lat_col].to_numpy(dtype=np.float32)
        ts  = g[time_col].values.astype("datetime64[s]").astype(np.float64)
        dt  = np.concatenate([[0.0], np.diff(ts)]).astype(np.float32)
        pts = np.stack([lon, lat, dt], axis=1)
        if pts.shape[0] >= 5:
            key = str(v_id)
            tracks[key] = pts
            if vessel_type_col in g.columns:
                vt = g[vessel_type_col].dropna()
                vessel_types[key] = int(vt.iloc[0]) if len(vt) > 0 else 0
            else:
                vessel_types[key] = 0

    lengths = [v.shape[0] for v in tracks.values()]
    print(f"Loaded {len(tracks):,} vessels")
    print(f"  Track length -- min:{min(lengths)}  "
          f"median:{np.median(lengths):.0f}  max:{max(lengths)}")
    return tracks, vessel_types


def train_val_split(
    tracks: Dict[str, np.ndarray],
    vessel_types: Dict[str, int],
    val_frac: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int],
           Dict[str, np.ndarray], Dict[str, int]]:
    ids = list(tracks.keys())
    random.Random(seed).shuffle(ids)
    if len(ids) <= 1 or val_frac <= 0:
        return tracks, vessel_types, {}, {}
    n_val   = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    return (
        {k: v for k, v in tracks.items()       if k not in val_ids},
        {k: v for k, v in vessel_types.items() if k not in val_ids},
        {k: v for k, v in tracks.items()       if k in val_ids},
        {k: v for k, v in vessel_types.items() if k in val_ids},
    )


# ─── Feature building ─────────────────────────────────────────────────────────

def make_input(pts: np.ndarray, scalers: Scalers, use_velocity: bool) -> np.ndarray:
    """
    Build normalised input features from a (T, 3) trajectory window.
    Returns (T, 3) or (T, 5).
    Velocity features use disp scaler so they are the same scale as the target.
    """
    pos_norm   = scalers.pos.transform(pts[:, :2])
    logdt_norm = scalers.logdt.transform(np.log1p(pts[:, 2]).reshape(-1, 1))
    x = np.concatenate([pos_norm, logdt_norm], axis=1)
    if use_velocity:
        dpos     = np.diff(pts[:, :2], axis=0, prepend=pts[0:1, :2])
        vel_norm = scalers.disp.transform(dpos)   # disp-normalized: same scale as target
        x = np.concatenate([x, vel_norm], axis=1)
    return x


# ─── Datasets ─────────────────────────────────────────────────────────────────

class CausalDataset(Dataset):
    """
    Seq2seq dataset with causal mask.

    x         : (seq_len, input_dim)  — input positions 0..seq_len-1
    y         : (seq_len, 2)          — normalized displacement to next position at each step
    gap_mask  : (seq_len,) bool       — True where dt > max_gap_sec (masked in attention)
    vtype_idx : int64 scalar          — vessel type vocab index

    At inference, read off y[-1] as the 1-step-ahead displacement prediction.
    """

    def __init__(self, tracks: Dict[str, np.ndarray],
                 vessel_types: Dict[str, int],
                 vtype_vocab: Dict[int, int],
                 scalers: Scalers,
                 seq_len: int, stride: int = 1,
                 max_windows_per_track: Optional[int] = None,
                 use_velocity: bool = True, max_gap_sec: float = 600):
        self.tracks       = tracks
        self.vessel_types = vessel_types
        self.vtype_vocab  = vtype_vocab
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
        pts = self.tracks[v_id][s : s + self.seq_len + 1]   # (seq_len+1, 3)

        x        = make_input(pts[:-1], self.scalers, self.use_velocity)
        disp_raw = pts[1:, :2] - pts[:-1, :2]               # (seq_len, 2)
        y        = self.scalers.disp.transform(disp_raw)

        dt       = pts[:-1, 2]
        gap_mask = torch.tensor(dt > self.max_gap_sec, dtype=torch.bool)

        raw_code  = self.vessel_types.get(v_id, 0)
        vtype_idx = torch.tensor(self.vtype_vocab.get(raw_code, 0), dtype=torch.long)

        return torch.tensor(x), torch.tensor(y), gap_mask, vtype_idx


class SingleStepDataset(Dataset):
    """
    Bidirectional single-step dataset.

    x         : (seq_len, input_dim)  — full context window (no leakage: y is not in x)
    y         : (2,)                  — normalized displacement for the step after the window
    gap_mask  : (seq_len,) bool
    vtype_idx : int64 scalar
    """

    def __init__(self, tracks: Dict[str, np.ndarray],
                 vessel_types: Dict[str, int],
                 vtype_vocab: Dict[int, int],
                 scalers: Scalers,
                 seq_len: int, stride: int = 1,
                 max_windows_per_track: Optional[int] = None,
                 use_velocity: bool = True, max_gap_sec: float = 600):
        self.tracks       = tracks
        self.vessel_types = vessel_types
        self.vtype_vocab  = vtype_vocab
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

        x        = make_input(pts[:-1], self.scalers, self.use_velocity)
        disp_raw = pts[-1:, :2] - pts[-2:-1, :2]            # (1, 2)
        y        = self.scalers.disp.transform(disp_raw)[0]  # (2,)

        dt       = pts[:-1, 2]
        gap_mask = torch.tensor(dt > self.max_gap_sec, dtype=torch.bool)

        raw_code  = self.vessel_types.get(v_id, 0)
        vtype_idx = torch.tensor(self.vtype_vocab.get(raw_code, 0), dtype=torch.long)

        return torch.tensor(x), torch.tensor(y), gap_mask, vtype_idx


def causal_collate(batch):
    xs, ys, gms, vtypes = zip(*batch)
    return (
        torch.stack(xs).transpose(0, 1),        # (seq_len, B, D)
        torch.stack(ys).transpose(0, 1),        # (seq_len, B, 2)
        torch.stack(gms),                        # (B, seq_len)
        torch.stack(vtypes),                     # (B,) int64
    )


def single_collate(batch):
    xs, ys, gms, vtypes = zip(*batch)
    return (
        torch.stack(xs).transpose(0, 1),        # (seq_len, B, D)
        torch.stack(ys),                         # (B, 2)
        torch.stack(gms),                        # (B, seq_len)
        torch.stack(vtypes),                     # (B,) int64
    )


def get_loader(
    tracks: Dict[str, np.ndarray],
    vessel_types: Dict[str, int],
    vtype_vocab: Dict[int, int],
    scalers: Scalers,
    seq_len: int,
    batch_size: int,
    pred_mode: str = "causal",
    stride: int = 1,
    max_windows_per_track: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    use_velocity: bool = True,
    num_workers: int = 0,
    max_gap_sec: float = 600,
) -> DataLoader:
    if pred_mode == "causal":
        dataset = CausalDataset(tracks, vessel_types, vtype_vocab, scalers,
                                seq_len, stride, max_windows_per_track,
                                use_velocity, max_gap_sec)
        collate = causal_collate
    else:
        dataset = SingleStepDataset(tracks, vessel_types, vtype_vocab, scalers,
                                    seq_len, stride, max_windows_per_track,
                                    use_velocity, max_gap_sec)
        collate = single_collate

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=False)
