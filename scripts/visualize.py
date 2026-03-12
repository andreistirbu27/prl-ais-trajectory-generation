#!/usr/bin/env python3
"""
visualize.py — AIS data distribution and model predictions.

Produces two plots:
  viz_distribution.png  — train/val vessel coverage on the US coast
  viz_predictions.png   — model vs constant-velocity vs truth for sample val vessels
                          (requires --checkpoint)

Usage:
    # Distribution only:
    python3 scripts/visualize.py --csv data/processed/AIS_combined_processed.csv

    # With model predictions:
    python3 scripts/visualize.py \
        --csv data/processed/AIS_combined_processed.csv \
        --checkpoint runs/ais_transformer/best.pt
"""

import argparse
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import Scaler, Scalers, load_tracks, make_input, train_val_split
from src.model import AISTransformer


# ── vessel type labels ────────────────────────────────────────────────────────

VTYPE_NAME = {
    **{c: "Passenger" for c in range(60, 70)},
    **{c: "Cargo"     for c in range(70, 80)},
    **{c: "Tanker"    for c in range(80, 90)},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    scalers = Scalers(
        pos   = Scaler(mean=ckpt["pos_mean"],   std=ckpt["pos_std"]),
        logdt = Scaler(mean=ckpt["logdt_mean"], std=ckpt["logdt_std"]),
        disp  = Scaler(mean=ckpt["disp_mean"],  std=ckpt["disp_std"]),
    )
    input_dim = 5 if a.get("use_velocity", True) else 3
    model = AISTransformer(
        input_dim=input_dim,
        d_model=a.get("d_model", 128),
        nhead=a.get("nhead", 8),
        num_layers=a.get("num_layers", 2),
        dropout=0.0,
        pred_mode=a.get("pred_mode", "causal"),
        num_vessel_types=ckpt.get("num_vessel_types", 0),
        vessel_type_embed_dim=8,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, scalers, ckpt.get("vtype_vocab", {0: 0}), a, ckpt.get("epoch", "?")


# ── Figure 1: data distribution ───────────────────────────────────────────────

def plot_distribution(train_tracks, val_tracks, vessel_types, out_path):
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_facecolor("#cce0f0")

    def sample_points(tracks, max_per_vessel=60):
        lons, lats = [], []
        for pts in tracks.values():
            step = max(1, len(pts) // max_per_vessel)
            lons.extend(pts[::step, 0].tolist())
            lats.extend(pts[::step, 1].tolist())
        return np.array(lons), np.array(lats)

    tr_lon, tr_lat = sample_points(train_tracks)
    va_lon, va_lat = sample_points(val_tracks)

    ax.scatter(tr_lon, tr_lat, s=0.3, alpha=0.35, color="#1565c0", rasterized=True,
               label=f"Train  ({len(train_tracks):,} vessels, {len(tr_lon):,} pts shown)")
    ax.scatter(va_lon, va_lat, s=0.8, alpha=0.7,  color="#d84315", rasterized=True,
               label=f"Val    ({len(val_tracks):,} vessels, {len(va_lon):,} pts shown)")

    # vessel type breakdown annotation
    counts: dict = {}
    for code in vessel_types.values():
        name = VTYPE_NAME.get(code, f"Code {code}")
        counts[name] = counts.get(name, 0) + 1
    info = "   ".join(f"{k}: {v:,}" for k, v in sorted(counts.items()))
    ax.text(0.01, 0.02, info, transform=ax.transAxes, fontsize=9,
            color="white", bbox=dict(fc="#000000aa", boxstyle="round,pad=0.3"))

    ax.set_xlabel("Longitude (°)", fontsize=11)
    ax.set_ylabel("Latitude (°)",  fontsize=11)
    ax.set_title("AIS vessel track coverage — train vs validation split\n"
                 "(each dot = one ping; blue = training vessels, red = validation vessels)",
                 fontsize=13)
    ax.legend(markerscale=12, fontsize=10, loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 2: prediction panels ───────────────────────────────────────────────

@torch.no_grad()
def plot_predictions(model, val_tracks, val_vtypes, vtype_vocab, scalers,
                     model_args, device, n_vessels, out_path, seed=42):
    rng       = random.Random(seed)
    seq_len   = model_args.get("seq_len", 120)
    use_vel   = model_args.get("use_velocity", True)
    pred_mode = model_args.get("pred_mode", "causal")
    gap_sec   = model_args.get("max_gap_sec", 600.0)

    eligible = [(vid, pts) for vid, pts in val_tracks.items() if len(pts) >= seq_len + 1]
    if not eligible:
        print("  No eligible val vessels for prediction plot.")
        return

    # try to pick geographically diverse vessels
    rng.shuffle(eligible)
    chosen = eligible[:n_vessels]

    ncols = 4
    nrows = math.ceil(len(chosen) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.2))
    axes_flat = np.array(axes).reshape(-1)

    for i, (vid, pts) in enumerate(chosen):
        ax = axes_flat[i]
        T = len(pts)

        # pick a window in the second half of the track (more interesting behaviour)
        lo = max(0, T // 2 - seq_len)
        hi = T - seq_len - 1
        start = rng.randint(lo, hi) if hi > lo else 0
        window = pts[start : start + seq_len + 1]   # (seq_len+1, 3)

        ctx  = window[:-1]          # (seq_len, 3) — context
        true_next = window[-1, :2]  # [lon, lat]
        last_pos  = ctx[-1, :2]

        # model prediction
        x_feat = make_input(ctx, scalers, use_vel)                   # (seq_len, D)
        x_t    = torch.tensor(x_feat).unsqueeze(1).to(device)        # (seq_len,1,D)
        dt     = ctx[:, 2]
        gm     = torch.tensor(dt > gap_sec, dtype=torch.bool).unsqueeze(0).to(device)
        raw_code  = val_vtypes.get(vid, 0)
        vt_idx = torch.tensor([vtype_vocab.get(raw_code, 0)], dtype=torch.long).to(device)

        out = model(x_t, gap_mask=gm, vessel_type=vt_idx)
        disp_norm = out[-1, 0].cpu().numpy() if pred_mode == "causal" else out[0].cpu().numpy()
        pred_disp = scalers.disp.inverse(disp_norm.reshape(1, 2))[0]
        model_next = last_pos + pred_disp

        # constant-velocity baseline
        cv_disp = ctx[-1, :2] - ctx[-2, :2] if len(ctx) >= 2 else np.zeros(2)
        cv_next = last_pos + cv_disp

        err_model = haversine_m(true_next[1], true_next[0], model_next[1], model_next[0])
        err_cv    = haversine_m(true_next[1], true_next[0], cv_next[1],    cv_next[0])

        # ── draw ──
        lons = ctx[:, 0]
        lats = ctx[:, 1]

        # colour context by age (older = lighter)
        n_seg = len(lons) - 1
        cmap  = plt.cm.Blues
        for j in range(n_seg):
            alpha = 0.3 + 0.7 * (j / max(n_seg - 1, 1))
            ax.plot(lons[j:j+2], lats[j:j+2], color=cmap(0.4 + 0.6 * j / max(n_seg-1, 1)),
                    linewidth=1.0, alpha=alpha)

        # sampled dots along context
        step = max(1, len(lons) // 15)
        ax.scatter(lons[::step], lats[::step], s=6, color="#1565c0", alpha=0.6, zorder=3)

        # last known position
        ax.scatter(*last_pos, s=70, color="navy", zorder=6, marker="o")

        # true next
        ax.scatter(*true_next, s=150, color="#2e7d32", zorder=8, marker="*")
        # model prediction
        ax.scatter(*model_next, s=90,  color="#c62828", zorder=7, marker="^")
        # cv baseline
        ax.scatter(*cv_next,   s=70,  color="#e65100", zorder=7, marker="D")

        # arrows from last known position
        def arrow(src, dst, color):
            ax.annotate("", xy=dst, xytext=src,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, alpha=0.8))

        arrow(last_pos, true_next,  "#2e7d32")
        arrow(last_pos, model_next, "#c62828")
        arrow(last_pos, cv_next,    "#e65100")

        vname = VTYPE_NAME.get(raw_code, f"Type {raw_code}")
        ax.set_title(f"MMSI {vid}  [{vname}]", fontsize=8, pad=3)

        # compact legend via proxy artists
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#2e7d32", markersize=9,  label="True next"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#c62828", markersize=7,  label=f"Model  {err_model:.0f}m"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#e65100", markersize=6,  label=f"CV     {err_cv:.0f}m"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Lon", fontsize=7)
        ax.set_ylabel("Lat", fontsize=7)

    for ax in axes_flat[len(chosen):]:
        ax.set_visible(False)

    fig.suptitle(
        "Model predictions on validation vessels\n"
        "Blue line = context window (darkens toward present)  |  "
        "● = last known pos  |  ★ = true next  |  ▲ = model  |  ◆ = const-vel baseline",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv",        required=True,             help="Processed AIS CSV")
    p.add_argument("--checkpoint", default=None,              help="Path to best.pt (optional)")
    p.add_argument("--val_frac",   type=float, default=0.15,  help="Must match training val_frac")
    p.add_argument("--seed",       type=int,   default=42,    help="Must match training seed")
    p.add_argument("--n_vessels",  type=int,   default=12,    help="Number of prediction panels")
    p.add_argument("--out_dir",    default="runs/ais_transformer")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    print("Loading data...")
    tracks, vessel_types = load_tracks(args.csv)
    train_tracks, train_vtypes, val_tracks, val_vtypes = train_val_split(
        tracks, vessel_types, args.val_frac, args.seed)
    print(f"  Train: {len(train_tracks):,}  Val: {len(val_tracks):,}")

    print("\n[1/2] Plotting data distribution...")
    plot_distribution(train_tracks, val_tracks, vessel_types,
                      os.path.join(args.out_dir, "viz_distribution.png"))

    if args.checkpoint:
        print(f"\n[2/2] Loading checkpoint: {args.checkpoint}")
        model, scalers, vtype_vocab, model_args, epoch = load_checkpoint(args.checkpoint, device)
        print(f"      Epoch {epoch}")
        plot_predictions(model, val_tracks, val_vtypes, vtype_vocab, scalers,
                         model_args, device, args.n_vessels,
                         os.path.join(args.out_dir, "viz_predictions.png"),
                         seed=args.seed)
    else:
        print("\n[2/2] No --checkpoint given — skipping prediction plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
