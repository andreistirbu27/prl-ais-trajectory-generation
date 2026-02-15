# CSV columns expected by default: MMSI, BaseDateTime, LAT, LON

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
    mean: np.ndarray  # (2,)
    std: np.ndarray   # (2,)

    @staticmethod
    def fit(xy: np.ndarray) -> "Scaler2D":
        mean = xy.mean(axis=0)
        std = xy.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return Scaler2D(mean=mean, std=std)

    def transform(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.mean) / self.std

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        return xy * self.std + self.mean


# AIS data loading

def load_tracks(
    csv_path: str,
    id_col: str = "MMSI",
    time_col: str = "BaseDateTime",
    lat_col: str = "LAT",
    lon_col: str = "LON",
) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)

    tracks: Dict[str, np.ndarray] = {}
    for v_id, g in df.groupby(id_col):
        lon = g[lon_col].astype(float).to_numpy()
        lat = g[lat_col].astype(float).to_numpy()
        pts = np.stack([lon, lat], axis=1).astype(np.float32)  # (T,2)
        if pts.shape[0] >= 5:
            tracks[str(v_id)] = pts
    return tracks


def train_val_split(tracks: Dict[str, np.ndarray], val_frac: float, seed: int):
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
    """
    Sliding windows from per-vessel trajectories.

    sample["source_sequence"]: (seq_len, 2) normalized [lon, lat]
    sample["target"]:          (seq_len, 2) normalized [lon, lat] shifted by 1 step
    """

    def __init__(
        self,
        tracks: Dict[str, np.ndarray],
        scaler: Scaler2D,
        seq_len: int = 20,
        stride: int = 1,
        max_windows_per_track: Optional[int] = None,
    ):
        self.tracks = tracks
        self.scaler = scaler
        self.seq_len = seq_len
        self.stride = stride
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
        pts = self.tracks[v_id][s : s + self.seq_len + 1]
        x = pts[:-1].astype(np.float64)
        y = pts[1:].astype(np.float64)
        x = self.scaler.transform(x).astype(np.float32)
        y = self.scaler.transform(y).astype(np.float32)
        return {"source_sequence": torch.tensor(x), "target": torch.tensor(y)}


def MyCollator(batch):
    x = torch.stack([sample["source_sequence"] for sample in batch], dim=0)  # (batch_size,seq_len,2)
    y = torch.stack([sample["target"] for sample in batch], dim=0)           # (batch_size,seq_len,2)
    return x.transpose(0, 1), y.transpose(0, 1)  # (seq_len,batch_size,2), (seq_len,batch_size,2)


def get_loader(
    tracks: Dict[str, np.ndarray],
    scaler: Scaler2D,
    seq_len: int,
    batch_size: int,
    stride: int = 1,
    max_windows_per_track: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
):
    dataset = TrajectoryDataset(
        tracks=tracks,
        scaler=scaler,
        seq_len=seq_len,
        stride=stride,
        max_windows_per_track=max_windows_per_track,
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
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        nhid: int,
        nlayers: int,
        dropout: float = 0.1,
    ):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)

        # dim_feedforward set to nhid (this can be made 4*nhid later)
        encoder_layers = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)

        self.nhid = nhid
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
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

    def forward(self, src, src_mask):
        # src: (seq_len,batch_size,2) -> (seq_len,batch_size,nhid)
        src = self.encoder(src) * math.sqrt(self.nhid)
        if self.pos_encoder is not None:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output


class RegressionHead(nn.Module):
    def __init__(self, nhid: int, out_dim: int = 2):
        super(RegressionHead, self).__init__()
        self.decoder = nn.Linear(nhid, out_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.decoder(src)
        return output


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        nhid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super(Model, self).__init__()
        self.base = TransformerModel(
            input_dim=input_dim,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout,
        )
        self.regressor = RegressionHead(nhid, out_dim=2)

    def forward(self, src, src_mask):
        x = self.base(src, src_mask)
        output = self.regressor(x)
        return output


# Metrics + baseline

def haversine_meters(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371000.0 # Earth's Radius
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
def eval_regression(model: Model, loader: DataLoader, device: torch.device, scaler: Scaler2D):
    model.eval()
    mse_fn = nn.MSELoss(reduction="mean")


    total_mse = 0.0
    n_batches = 0

    ade_sum = 0.0
    fde_sum = 0.0

    for src, target in loader:
        src = src.to(device)
        target = target.to(device)
        src_mask = model.base.generate_square_subsequent_mask(src.size(0)).to(device)
        out = model(src, src_mask)

        total_mse += float(mse_fn(out, target).item())
        n_batches += 1

        out_np = out.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()

        out_ll = scaler.inverse(out_np.reshape(-1, 2)).reshape(out_np.shape)   # lon,lat
        tgt_ll = scaler.inverse(tgt_np.reshape(-1, 2)).reshape(tgt_np.shape)

        dist = haversine_meters(
            lat1=tgt_ll[:, :, 1], lon1=tgt_ll[:, :, 0],
            lat2=out_ll[:, :, 1], lon2=out_ll[:, :, 0],
        )
        ade_sum += float(dist.mean())

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
    mse_fn = nn.MSELoss(reduction="mean")
    total_mse = 0.0
    n_batches = 0

    ade_sum = 0.0
    fde_sum = 0.0

    for src, target in loader:
        src = src.to(device)
        target = target.to(device)

        pred = torch.zeros_like(target)
        pred[0] = src[0]
        pred[1:] = src[1:] + (src[1:] - src[:-1])

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
):
    model.train()
    total_loss = 0.0
    mse_fn = nn.MSELoss(reduction="mean")

    for idx, data in enumerate(loader):
        optimizer.zero_grad()

        src, target = data  # (seq_len,batch_size,2), (seq_len,batch_size,2)
        src_mask = model.base.generate_square_subsequent_mask(src.size(0)).to(device)

        src = src.to(device)
        target = target.to(device)

        output = model(src, src_mask)  # (seq_len,batch_size,2)

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

    parser = argparse.ArgumentParser(description="TP6-style Transformer adapted to AIS next-point prediction.")
    parser.add_argument("--csv", required=True, help="Path to AIS CSV file.")
    parser.add_argument("--out_dir", default="runs/ais_transformer_tp6_style")

    parser.add_argument("--id_col", default="MMSI")
    parser.add_argument("--time_col", default="BaseDateTime")
    parser.add_argument("--lat_col", default="LAT")
    parser.add_argument("--lon_col", default="LON")

    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_windows_per_track", type=int, default=None)

    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_positional_encoding", action="store_true")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.5)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    tracks = load_tracks(
        args.csv,
        id_col=args.id_col,
        time_col=args.time_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
    )
    print(f"Loaded {len(tracks)} vessels")

    train_tracks, val_tracks = train_val_split(tracks, args.val_frac, args.seed)
    print(f"Split: {len(train_tracks)} train vessels, {len(val_tracks)} val vessels")

    if len(train_tracks) == 0:
        raise RuntimeError("No training tracks. Check CSV columns and data.")

    all_train_pts = np.concatenate(list(train_tracks.values()), axis=0).astype(np.float64)  # lon,lat
    scaler = Scaler2D.fit(all_train_pts)

    train_loader = get_loader(
        tracks=train_tracks,
        scaler=scaler,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        stride=args.stride,
        max_windows_per_track=args.max_windows_per_track,
        shuffle=True,
        drop_last=True,
    )

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
        )

    model = Model(
        input_dim=2,
        nhead=args.nhead,
        nhid=args.nhid,
        nlayers=args.nlayers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)

    best_val = float("inf")

    if val_loader is not None:
        base = eval_constant_velocity_baseline(val_loader, device, scaler)
        print(f"Baseline (const vel) | val mse {base['mse']:.6f} | ADE {base['ade_m']:.1f} m | FDE {base['fde_m']:.1f} m")

    for epoch in range(1, args.epochs + 1):
        train(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            log_interval=20,
            current_epoch=epoch,
            grad_clip=args.grad_clip,
        )

        if val_loader is not None:
            metrics = eval_regression(model, val_loader, device, scaler)
            print(
                f"Eval | epoch {epoch:3d} | val mse {metrics['mse']:.6f} | "
                f"ADE {metrics['ade_m']:.1f} m | FDE {metrics['fde_m']:.1f} m"
            )

            if metrics["mse"] < best_val:
                best_val = metrics["mse"]
                torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
                np.save(os.path.join(args.out_dir, "scaler_mean.npy"), scaler.mean)
                np.save(os.path.join(args.out_dir, "scaler_std.npy"), scaler.std)
                print(f"Saved checkpoint to {args.out_dir}")
        else:
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

    print("Done.")


if __name__ == "__main__":
    main()
