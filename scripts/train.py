#!/usr/bin/env python3
"""
train.py — AIS Vessel Trajectory Prediction

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
  [lon_norm, lat_norm, log_dt_norm]                         (3-dim, default)
  [lon_norm, lat_norm, log_dt_norm, dlon_norm, dlat_norm]  (5-dim, --use_velocity)

Target: displacement to next position [dlon_norm, dlat_norm]
  Predicted next position = input position + disp_scaler.inverse(pred)
  This is much easier to learn than absolute position because the target
  variance is small (centered near zero) regardless of where on the coast
  the vessel is.

Usage:
    python3 scripts/train.py --csv data/processed/AIS_combined_processed.csv
    python3 scripts/train.py --csv data/processed/AIS_combined_processed.csv \\
        --pred_mode single --seq_len 30 --d_model 256 --num_layers 4
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import (Scalers, load_tracks, train_val_split, get_loader)
from src.model import AISTransformer
from src.metrics import (evaluate, evaluate_constant_velocity_baseline, sanity_check)


# ── Reproducibility + device ──────────────────────────────────────────────────

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


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler,
                    device, grad_clip, epoch, pred_mode, log_every=50,
                    lambda_smooth=0.0, loss_fn="mse"):
    model.train()
    base_loss_fn = nn.HuberLoss(delta=1.0) if loss_fn == "huber" else nn.MSELoss()
    running, running_smooth, total, n = 0.0, 0.0, 0.0, 0

    for i, (x, y, gap_mask, vtype) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        gap_mask = gap_mask.to(device)
        vtype    = vtype.to(device)
        optimizer.zero_grad()
        pred = model(x, gap_mask=gap_mask, vessel_type=vtype)

        if pred_mode == "causal":
            # Skip t=0: velocity feature is always 0 there (diff prepend artifact),
            # which pairs the same input with wildly different targets → noisy gradient.
            mse_loss = base_loss_fn(pred[1:], y[1:])
            smooth_loss = torch.tensor(0.0, device=device)
            if lambda_smooth > 0 and pred.size(0) > 2:
                # Penalise acceleration: change in predicted displacement per step
                accel = pred[2:] - pred[1:-1]          # (seq_len-2, B, 2)
                smooth_loss = (accel ** 2).mean()
            loss = mse_loss + lambda_smooth * smooth_loss
        else:
            mse_loss = base_loss_fn(pred, y)
            smooth_loss = torch.tensor(0.0, device=device)
            loss = mse_loss

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        running        += mse_loss.item()
        running_smooth += smooth_loss.item()
        total          += mse_loss.item()
        n              += 1

        if (i + 1) % log_every == 0:
            smooth_str = (f" | smooth {running_smooth/log_every:.4f}"
                          if lambda_smooth > 0 else "")
            print(f"  epoch {epoch:3d} | step {i+1:5d}/{len(loader)} "
                  f"| mse {running/log_every:.6f}{smooth_str} "
                  f"| grad {grad_norm:.3f} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e}")
            running = 0.0
            running_smooth = 0.0

    return total / max(n, 1)


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
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
    parser.add_argument("--use_velocity",          action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val_frac",              type=float, default=0.1)
    parser.add_argument("--seed",                  type=int,   default=42)

    parser.add_argument("--d_model",         type=int,   default=128)
    parser.add_argument("--nhead",           type=int,   default=8)
    parser.add_argument("--num_layers",      type=int,   default=2)
    parser.add_argument("--dim_feedforward", type=int,   default=None)
    parser.add_argument("--dropout",         type=float, default=0.1)

    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--warmup_frac",    type=float, default=0.05)
    parser.add_argument("--num_workers",    type=int,   default=0)
    parser.add_argument("--lambda_smooth",  type=float, default=0.1,
                        help="Weight for speed-smoothness regularisation "
                             "(MSE + lambda*accel^2). 0=off.")
    parser.add_argument("--loss_fn",        default="mse", choices=["mse", "huber"],
                        help="Base loss function. huber is more robust to outliers.")

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

    tracks, vessel_types = load_tracks(args.csv, args.id_col, args.time_col,
                                       args.lat_col, args.lon_col)
    if not tracks:
        raise RuntimeError("No valid tracks loaded.")

    train_tracks, train_vtypes, val_tracks, val_vtypes = train_val_split(
        tracks, vessel_types, args.val_frac, args.seed)
    print(f"  Train: {len(train_tracks):,} vessels  |  Val: {len(val_tracks):,}\n")

    scalers = Scalers.fit(train_tracks)
    print(f"  Position scaler -- mean: {scalers.pos.mean}  std: {scalers.pos.std}")
    print(f"  log(dt) scaler  -- mean: {scalers.logdt.mean.item():.4f}  "
          f"std: {scalers.logdt.std.item():.4f}")
    print(f"  Disp scaler     -- mean: {scalers.disp.mean}  std: {scalers.disp.std}\n")

    # Build vessel type vocab from training vessels only (1-indexed; 0 = unknown)
    all_codes   = sorted(set(train_vtypes.values()))
    vtype_vocab = {code: idx + 1 for idx, code in enumerate(all_codes)}
    vtype_vocab[0] = 0   # unknown / missing
    num_vessel_types = len(vtype_vocab)
    print(f"  Vessel types    : {num_vessel_types - 1} unique codes  "
          f"(vocab size {num_vessel_types}  codes: {all_codes})\n")

    input_dim = 5 if args.use_velocity else 3
    print(f"  Input  : {input_dim}D "
          f"{'[lon,lat,log_dt,vel_lon,vel_lat]' if args.use_velocity else '[lon,lat,log_dt]'}"
          f" + vessel-type embedding ({num_vessel_types} types → 8-dim)")
    print(f"  Target : displacement [dlon_norm, dlat_norm] (next pos = input pos + pred)")
    print(f"  Mode   : {args.pred_mode}")
    print(f"  Loss   : {args.loss_fn.upper()} + smoothness*{args.lambda_smooth} "
          f"(t=0 excluded from causal loss)\n")

    train_loader = get_loader(
        train_tracks, train_vtypes, vtype_vocab, scalers, args.seq_len, args.batch_size,
        pred_mode=args.pred_mode, stride=args.stride,
        max_windows_per_track=args.max_windows_per_track,
        shuffle=True, drop_last=True, use_velocity=args.use_velocity,
        num_workers=args.num_workers, max_gap_sec=args.max_gap_sec,
    )
    val_loader = get_loader(
        val_tracks, val_vtypes, vtype_vocab, scalers, args.seq_len, args.batch_size,
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
        num_vessel_types=num_vessel_types, vessel_type_embed_dim=8,
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
            lambda_smooth=args.lambda_smooth, loss_fn=args.loss_fn,
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
                    "disp_mean": scalers.disp.mean, "disp_std": scalers.disp.std,
                    "vtype_vocab": vtype_vocab, "num_vessel_types": num_vessel_types,
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
