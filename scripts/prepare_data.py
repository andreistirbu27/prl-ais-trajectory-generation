#!/usr/bin/env python3
"""
prepare_data.py — Combine raw AIS day files and filter to a clean dataset.

Accepts one or more input CSV files:
  - Multiple files: concatenate + deduplicate first, then filter
  - Single file: filter directly

Produces a cleaned CSV at data/processed/<stem>_processed.csv (or --output).

Usage:
    # Combine multiple days then filter:
    python3 scripts/prepare_data.py data/raw/AIS_2024_03_26.csv \\
        data/raw/AIS_2024_03_27.csv data/raw/AIS_2024_03_28.csv

    # Filter an already-combined file:
    python3 scripts/prepare_data.py data/raw/AIS_mar.csv

    # Audit without writing output (dry-run):
    python3 scripts/prepare_data.py data/raw/AIS_mar.csv --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Audit log ─────────────────────────────────────────────────────────────────

_steps: list = []
_initial_rows: int = 0


def _record(step: str, before: int, after: int, detail: str = ""):
    removed = before - after
    pct = 100 * removed / _initial_rows if _initial_rows > 0 else 0
    _steps.append(dict(step=step, before=before, after=after,
                       removed=removed, pct=pct, detail=detail))


def _print_report(df: pd.DataFrame, id_col: str, time_col: str,
                  lat_col: str, lon_col: str, out_path, dry_run: bool = False):
    W = 70
    print("=" * W)
    print("  AIS PREPROCESSING AUDIT LOG")
    print("=" * W)

    print(f"\n{'Step':<35} {'Removed':>9}  {'% of total':>10}  {'Remaining':>10}")
    print("-" * W)
    for s in _steps:
        detail = f"  ({s['detail']})" if s["detail"] else ""
        print(f"{s['step']:<35} {s['removed']:>9,}  {s['pct']:>9.2f}%  {s['after']:>10,}{detail}")

    total_removed = _initial_rows - len(df)
    total_pct = 100 * total_removed / _initial_rows if _initial_rows > 0 else 0
    print("-" * W)
    print(f"{'TOTAL REMOVED':<35} {total_removed:>9,}  {total_pct:>9.2f}%  {len(df):>10,}")

    print(f"\n{'─' * W}")
    print("  OUTPUT SUMMARY")
    print(f"{'─' * W}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Vessels: {df[id_col].nunique():,}")

    track_len = df.groupby(id_col).size()
    print(f"  Points per vessel  — min: {track_len.min()}  "
          f"median: {track_len.median():.0f}  max: {track_len.max()}")

    dt_sec = df.groupby(id_col)[time_col].diff().dt.total_seconds().dropna()
    print(f"  Temporal gap (sec) — mean: {dt_sec.mean():.0f}  "
          f"median: {dt_sec.median():.0f}  p95: {dt_sec.quantile(0.95):.0f}")

    dlon = df.groupby(id_col)[lon_col].diff()
    dlat = df.groupby(id_col)[lat_col].diff()
    jump_km = (np.sqrt(dlon**2 + dlat**2) * 111).dropna()
    print(f"  Jump distance (km) — mean: {jump_km.mean():.2f}  "
          f"median: {jump_km.median():.2f}  p95: {jump_km.quantile(0.95):.2f}")

    valid_dt_h = dt_sec[dt_sec > 0] / 3600
    speed = jump_km[dt_sec > 0] / valid_dt_h
    print(f"  Est. speed (km/h)  — mean: {speed.mean():.1f}  "
          f"median: {speed.median():.1f}  p95: {speed.quantile(0.95):.1f}")

    print(f"\n  Time range: {df[time_col].min()}  →  {df[time_col].max()}")
    print(f"  Lat range:  [{df[lat_col].min():.4f}, {df[lat_col].max():.4f}]")
    print(f"  Lon range:  [{df[lon_col].min():.4f}, {df[lon_col].max():.4f}]")

    if dry_run:
        print(f"\n  [dry-run — no file written]")
    else:
        print(f"\n  Saved → {out_path}")
    print("=" * W)


# ── Combine step ──────────────────────────────────────────────────────────────

def combine(inputs: list, id_col: str, time_col: str) -> pd.DataFrame:
    dfs = []
    for path in inputs:
        print(f"Loading {path} ...")
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
        print(f"  {len(df):,} rows, {df[id_col].nunique():,} vessels")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined):,} rows before dedup")
    combined = combined.drop_duplicates(subset=[id_col, time_col])
    print(f"Combined: {len(combined):,} rows after dedup")
    print(f"Unique vessels: {combined[id_col].nunique():,}\n")
    return combined


# ── Filter step ───────────────────────────────────────────────────────────────

def filter_data(df: pd.DataFrame, args) -> pd.DataFrame:
    global _initial_rows
    _steps.clear()

    id_col   = args.id_col
    time_col = args.time_col
    lat_col  = args.lat_col
    lon_col  = args.lon_col
    vtype_col = args.vessel_type_col
    cols = [id_col, time_col, lat_col, lon_col]

    _initial_rows = len(df)

    # 1. Missing values
    before = len(df)
    df = df.dropna(subset=cols)
    _record("1. Drop missing values", before, len(df))

    # 1b. Vessel type filter
    if args.keep_vessel_types and vtype_col in df.columns:
        before = len(df)
        allowed: set = set()
        for part in args.keep_vessel_types.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                allowed.update(range(int(lo), int(hi) + 1))
            else:
                allowed.add(int(part))
        df = df[df[vtype_col].isin(allowed)].reset_index(drop=True)
        _record("1b. Vessel type filtered", before, len(df),
                f"kept VesselType in {args.keep_vessel_types}")

    # 2. Parse timestamps
    before = len(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    _record("2. Unparseable timestamps", before, len(df))

    # 3. Invalid coordinates
    before = len(df)
    df = df[df[lat_col].between(-90, 90) & df[lon_col].between(-180, 180)]
    _record("3. Out-of-range coordinates", before, len(df),
            "LAT not in [-90,90] or LON not in [-180,180]")

    # 4. Bounding box
    before = len(df)
    lat_ok = (df[lat_col] >= args.lat_min) & (df[lat_col] <= args.lat_max)
    lon_ok = (df[lon_col] >= args.lon_min) & (df[lon_col] <= args.lon_max)
    df = df[lat_ok & lon_ok]
    _record("4. Outside bounding box", before, len(df),
            f"LON [{args.lon_min}, {args.lon_max}]  LAT [{args.lat_min}, {args.lat_max}]")

    # 4b. Border-truncated trajectories
    if args.bbox_border_deg > 0:
        before = len(df)
        df = df.sort_values([id_col, time_col]).reset_index(drop=True)
        first_last = df.groupby(id_col).agg(
            first_lat=(lat_col, "first"), last_lat=(lat_col, "last"),
            first_lon=(lon_col, "first"), last_lon=(lon_col, "last"),
        )
        near_border = (
            (first_last["first_lat"] - args.lat_min < args.bbox_border_deg) |
            (args.lat_max - first_last["first_lat"] < args.bbox_border_deg) |
            (first_last["first_lon"] - args.lon_min < args.bbox_border_deg) |
            (args.lon_max - first_last["first_lon"] < args.bbox_border_deg) |
            (first_last["last_lat"] - args.lat_min < args.bbox_border_deg) |
            (args.lat_max - first_last["last_lat"] < args.bbox_border_deg) |
            (first_last["last_lon"] - args.lon_min < args.bbox_border_deg) |
            (args.lon_max - first_last["last_lon"] < args.bbox_border_deg)
        )
        truncated_ids = near_border[near_border].index
        df = df[~df[id_col].isin(truncated_ids)].reset_index(drop=True)
        _record("4b. Border-truncated trajectories", before, len(df),
                f"start/end within {args.bbox_border_deg}° of bbox edge")

    # 5. Duplicate (vessel, timestamp) pairs
    before = len(df)
    df = df.drop_duplicates(subset=[id_col, time_col])
    _record("5. Duplicate (vessel, timestamp)", before, len(df))

    # Sort (required for all diff-based filters below)
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # 5b. GPS median filter
    if args.median_filter_window > 1:
        w = args.median_filter_window
        df[lon_col] = (df.groupby(id_col)[lon_col]
                         .transform(lambda x: x.rolling(w, center=True, min_periods=1).median()))
        df[lat_col] = (df.groupby(id_col)[lat_col]
                         .transform(lambda x: x.rolling(w, center=True, min_periods=1).median()))

    # 6. Stationary streaks
    if args.max_stationary_min > 0:
        before = len(df)
        dlon = df.groupby(id_col)[lon_col].diff()
        dlat = df.groupby(id_col)[lat_col].diff()
        dt_s = df.groupby(id_col)[time_col].diff().dt.total_seconds()
        mask = (dlon == 0) & (dlat == 0) & (dt_s > args.max_stationary_min * 60)
        df = df[~mask].reset_index(drop=True)
        _record("6. Stationary too long", before, len(df),
                f"zero movement > {args.max_stationary_min} min")
        df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # 7. Spatial jumps
    before = len(df)
    dlon = df.groupby(id_col)[lon_col].diff()
    dlat = df.groupby(id_col)[lat_col].diff()
    jump_deg = np.sqrt(dlon**2 + dlat**2)
    max_jump_deg = args.max_jump_km / 111.0
    df = df[(jump_deg <= max_jump_deg) | jump_deg.isna()].reset_index(drop=True)
    _record("7. Spatial jump too large", before, len(df),
            f"> {args.max_jump_km} km")

    # 8. Temporal gap segmentation
    if args.segment_on_gap:
        before = len(df)
        df = df.sort_values([id_col, time_col]).reset_index(drop=True)
        dt_s = df.groupby(id_col)[time_col].diff().dt.total_seconds().fillna(0)
        gap_start = (dt_s > args.max_gap_min * 60)
        seg_idx = gap_start.groupby(df[id_col]).cumsum().astype(int)
        df[id_col] = df[id_col].astype(str) + "_" + seg_idx.astype(str)
        n_segments = df[id_col].nunique()
        _record("8. Temporal gap → segmented tracks", before, len(df),
                f"gap > {args.max_gap_min} min → {n_segments:,} segments")
    else:
        before = len(df)
        dt_s = df.groupby(id_col)[time_col].diff().dt.total_seconds()
        df = df[(dt_s <= args.max_gap_min * 60) | dt_s.isna()].reset_index(drop=True)
        _record("8. Temporal gap too large", before, len(df),
                f"> {args.max_gap_min} min")

    # 9. Unrealistic speed
    before = len(df)
    dlon = df.groupby(id_col)[lon_col].diff()
    dlat = df.groupby(id_col)[lat_col].diff()
    jump_km = np.sqrt(dlon**2 + dlat**2) * 111
    dt_h = df.groupby(id_col)[time_col].diff().dt.total_seconds() / 3600
    speed = jump_km / dt_h.replace(0, np.nan)
    df = df[(speed <= args.max_speed_kmh) | speed.isna()].reset_index(drop=True)
    _record("9. Unrealistic speed", before, len(df),
            f"> {args.max_speed_kmh} km/h")

    # 10. Short tracks
    before = len(df)
    counts = df.groupby(id_col)[id_col].transform("count")
    df = df[counts >= args.min_points].reset_index(drop=True)
    _record("10. Track too short", before, len(df),
            f"< {args.min_points} points")

    # 10b. Minimum total distance traveled
    if args.min_total_dist_km > 0:
        before = len(df)
        dlon = df.groupby(id_col)[lon_col].diff()
        dlat = df.groupby(id_col)[lat_col].diff()
        step_km = np.sqrt(dlon**2 + dlat**2) * 111
        total_dist = step_km.groupby(df[id_col]).sum()
        far_enough = total_dist[total_dist >= args.min_total_dist_km].index
        df = df[df[id_col].isin(far_enough)].reset_index(drop=True)
        _record("10b. Total distance too short", before, len(df),
                f"< {args.min_total_dist_km} km total arc-length")

    # 11. Slow / moored vessels
    if args.min_avg_speed_kmh > 0:
        before = len(df)
        dlon = df.groupby(id_col)[lon_col].diff()
        dlat = df.groupby(id_col)[lat_col].diff()
        jump_km = np.sqrt(dlon**2 + dlat**2) * 111
        dt_h = df.groupby(id_col)[time_col].diff().dt.total_seconds() / 3600
        speed = jump_km / dt_h.replace(0, np.nan)
        avg_speed = speed.groupby(df[id_col]).mean()
        fast_enough = avg_speed[avg_speed >= args.min_avg_speed_kmh].index
        df = df[df[id_col].isin(fast_enough)].reset_index(drop=True)
        _record("11. Vessel avg speed too low", before, len(df),
                f"avg speed < {args.min_avg_speed_kmh} km/h across track")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combine raw AIS day files and/or filter to a clean dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+",
                        help="One or more raw AIS CSV files. Multiple files are combined first.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: data/processed/<stem>_processed.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print audit log without writing any output file")

    # Column names
    parser.add_argument("--id_col",   default="MMSI")
    parser.add_argument("--time_col", default="BaseDateTime")
    parser.add_argument("--lat_col",  default="LAT")
    parser.add_argument("--lon_col",  default="LON")
    parser.add_argument("--vessel_type_col", default="VesselType")
    parser.add_argument("--keep_vessel_types", default="60-89",
                        help="Vessel type code ranges to keep (default: 60-89). "
                             "Empty string '' to disable.")

    # Filter thresholds
    parser.add_argument("--min_points",        type=int,   default=50)
    parser.add_argument("--max_jump_km",       type=float, default=2.0)
    parser.add_argument("--max_gap_min",       type=float, default=60.0)
    parser.add_argument("--max_speed_kmh",     type=float, default=40.0)
    parser.add_argument("--median_filter_window", type=int, default=3,
                        help="Rolling median window for GPS denoising (1 = disabled)")
    parser.add_argument("--max_stationary_min", type=float, default=5.0,
                        help="Drop identical consecutive positions > this many minutes (0 = off)")
    parser.add_argument("--segment_on_gap",    action=argparse.BooleanOptionalAction, default=True,
                        help="Split tracks at gaps instead of dropping pings (default: on)")
    parser.add_argument("--min_total_dist_km", type=float, default=5.0,
                        help="Min total arc-length per track in km (0 = off)")
    parser.add_argument("--bbox_border_deg",   type=float, default=0.1,
                        help="Drop trajectories starting/ending within this many degrees "
                             "of bbox edge (0 = off)")
    parser.add_argument("--min_avg_speed_kmh", type=float, default=1.0,
                        help="Min average speed per track in km/h (0 = off)")

    # Bounding box
    parser.add_argument("--lon_min", type=float, default=-125.0)
    parser.add_argument("--lon_max", type=float, default=-60.0)
    parser.add_argument("--lat_min", type=float, default=10.0)
    parser.add_argument("--lat_max", type=float, default=55.0)

    args = parser.parse_args()

    # Determine output path
    if len(args.inputs) == 1:
        inp = Path(args.inputs[0])
    else:
        inp = Path(args.inputs[0]).parent / ("_".join(
            Path(f).stem for f in args.inputs[:2]) + "_combined.csv")

    if args.output is None:
        out_path = Path("data") / "processed" / f"{inp.stem}_processed.csv"
    else:
        out_path = Path(args.output)

    # Load / combine
    if len(args.inputs) > 1:
        df = combine(args.inputs, args.id_col, args.time_col)
    else:
        print(f"Loading {args.inputs[0]} ...")
        all_cols = pd.read_csv(args.inputs[0], nrows=0).columns.tolist()
        cols = [args.id_col, args.time_col, args.lat_col, args.lon_col]
        load_cols = cols + ([args.vessel_type_col] if args.vessel_type_col in all_cols else [])
        try:
            df = pd.read_csv(args.inputs[0], usecols=load_cols)
        except ValueError as e:
            sys.exit(f"ERROR reading CSV: {e}")
        print(f"  {len(df):,} rows, {df[args.id_col].nunique():,} vessels\n")

    # Filter
    df = filter_data(df, args)

    # Write or skip
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    _print_report(df, args.id_col, args.time_col, args.lat_col, args.lon_col,
                  out_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
