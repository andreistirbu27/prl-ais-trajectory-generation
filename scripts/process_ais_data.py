#!/usr/bin/env python3


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Audit log
# ─────────────────────────────────────────────────────────────────────────────

_steps: list[dict] = []
_initial_rows: int = 0


def _record(step: str, before: int, after: int, detail: str = ""):
    removed = before - after
    pct = 100 * removed / _initial_rows if _initial_rows > 0 else 0
    _steps.append(dict(step=step, before=before, after=after,
                       removed=removed, pct=pct, detail=detail))


def _print_report(df: pd.DataFrame, id_col: str, time_col: str,
                  lat_col: str, lon_col: str, out_path: Path):
    W = 70
    print("=" * W)
    print("  AIS PREPROCESSING AUDIT LOG")
    print("=" * W)

    # Per-step table
    print(f"\n{'Step':<35} {'Removed':>9}  {'% of total':>10}  {'Remaining':>10}")
    print("-" * W)
    for s in _steps:
        detail = f"  ({s['detail']})" if s["detail"] else ""
        print(f"{s['step']:<35} {s['removed']:>9,}  {s['pct']:>9.2f}%  {s['after']:>10,}{detail}")

    # Overall
    total_removed = _initial_rows - len(df)
    total_pct = 100 * total_removed / _initial_rows if _initial_rows > 0 else 0
    print("-" * W)
    print(f"{'TOTAL REMOVED':<35} {total_removed:>9,}  {total_pct:>9.2f}%  {len(df):>10,}")

    # Dataset summary
    print(f"\n{'─' * W}")
    print("  OUTPUT SUMMARY")
    print(f"{'─' * W}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Vessels: {df[id_col].nunique():,}")

    track_len = df.groupby(id_col).size()
    print(f"  Points per vessel  — min: {track_len.min()}  "
          f"median: {track_len.median():.0f}  max: {track_len.max()}")

    dt_sec = (df.groupby(id_col)[time_col]
                .diff().dt.total_seconds().dropna())
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

    print(f"\n  Saved → {out_path}")
    print("=" * W)


def run(args):
    global _initial_rows

    id_col   = args.id_col
    time_col = args.time_col
    lat_col  = args.lat_col
    lon_col  = args.lon_col
    cols     = [id_col, time_col, lat_col, lon_col]

    inp = Path(args.input_csv)
    if args.output is None:
        out_path = Path("data") / "processed" / f"{inp.stem}_processed.csv"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(inp, usecols=cols)
    except ValueError as e:
        sys.exit(f"ERROR reading CSV: {e}")

    _initial_rows = len(df)

    # ── 1. Missing values ────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=cols)
    _record("1. Drop missing values", before, len(df))

    # ── 2. Parse timestamps ──────────────────────────────────────────────────
    before = len(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    _record("2. Unparseable timestamps", before, len(df))

    # ── 3. Invalid coordinates ───────────────────────────────────────────────
    before = len(df)
    df = df[df[lat_col].between(-90, 90) & df[lon_col].between(-180, 180)]
    _record("3. Out-of-range coordinates", before, len(df),
            "LAT not in [-90,90] or LON not in [-180,180]")

    # ── 4. Bounding box ──────────────────────────────────────────────────────
    before = len(df)
    lat_ok = (df[lat_col] >= args.lat_min) & (df[lat_col] <= args.lat_max)
    lon_ok = (df[lon_col] >= args.lon_min) & (df[lon_col] <= args.lon_max)
    df = df[lat_ok & lon_ok]
    _record("4. Outside bounding box", before, len(df),
            f"LON [{args.lon_min}, {args.lon_max}]  LAT [{args.lat_min}, {args.lat_max}]")

    # ── 5. Duplicate (vessel, timestamp) pairs ───────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=[id_col, time_col])
    _record("5. Duplicate (vessel, timestamp)", before, len(df))

    # ── Sort (required for all diff-based filters below) ────────────────────
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # ── 5. Stationary streaks ────────────────────────────────────────────────
    # Drop consecutive rows where the vessel hasn't moved AND the gap exceeds
    # --max_stationary_min. These are anchored vessels or stuck GPS reports.
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

    # ── 6. Spatial jumps ─────────────────────────────────────────────────────
    before = len(df)
    dlon = df.groupby(id_col)[lon_col].diff()
    dlat = df.groupby(id_col)[lat_col].diff()
    jump_deg = np.sqrt(dlon**2 + dlat**2)
    max_jump_deg = args.max_jump_km / 111.0
    df = df[(jump_deg <= max_jump_deg) | jump_deg.isna()].reset_index(drop=True)
    _record("7. Spatial jump too large", before, len(df),
            f"> {args.max_jump_km} km")

    # ── 7. Temporal gaps ─────────────────────────────────────────────────────
    before = len(df)
    dt_s = df.groupby(id_col)[time_col].diff().dt.total_seconds()
    df = df[(dt_s <= args.max_gap_min * 60) | dt_s.isna()].reset_index(drop=True)
    _record("8. Temporal gap too large", before, len(df),
            f"> {args.max_gap_min} min")

    # ── 8. Unrealistic speed ─────────────────────────────────────────────────
    before = len(df)
    dlon = df.groupby(id_col)[lon_col].diff()
    dlat = df.groupby(id_col)[lat_col].diff()
    jump_km = np.sqrt(dlon**2 + dlat**2) * 111
    dt_h = df.groupby(id_col)[time_col].diff().dt.total_seconds() / 3600
    speed = jump_km / dt_h.replace(0, np.nan)
    df = df[(speed <= args.max_speed_kmh) | speed.isna()].reset_index(drop=True)
    _record("9. Unrealistic speed", before, len(df),
            f"> {args.max_speed_kmh} km/h")

    # ── 9. Short vessel tracks ───────────────────────────────────────────────
    before = len(df)
    counts = df.groupby(id_col)[id_col].transform("count")
    df = df[counts >= args.min_points].reset_index(drop=True)
    _record("10. Track too short", before, len(df),
            f"< {args.min_points} points")

    # ── 10. Slow / moored vessels (avg speed across whole track) ────────────
    # Removes vessels that are anchored, moored, or drifting
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

    # ── Write & report ───────────────────────────────────────────────────────
    df.to_csv(out_path, index=False)
    _print_report(df, id_col, time_col, lat_col, lon_col, out_path)



def main():
    parser = argparse.ArgumentParser(
        description="Clean and filter a raw AIS CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_csv",
                        help="Path to raw AIS CSV")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: data/processed/<stem>_processed.csv)")

    # Column names
    parser.add_argument("--id_col",   default="MMSI")
    parser.add_argument("--time_col", default="BaseDateTime")
    parser.add_argument("--lat_col",  default="LAT")
    parser.add_argument("--lon_col",  default="LON")

    # Filter thresholds
    parser.add_argument("--min_points", type=int, default=20,
                        help="Min points per vessel track (default: 20)")
    parser.add_argument("--max_jump_km", type=float, default=5.0,
                        help="Max spatial jump between consecutive points in km "
                             "(default: 5.0)")
    parser.add_argument("--max_gap_min", type=float, default=60.0,
                        help="Max temporal gap between consecutive points in minutes "
                             "(default: 60)")
    parser.add_argument("--max_speed_kmh", type=float, default=80.0,
                        help="Max estimated speed between consecutive points in km/h "
                             "(default: 80)")
    parser.add_argument("--max_stationary_min", type=float, default=10.0,
                        help="Drop consecutive identical positions held longer than "
                             "this many minutes (default: 10). Set 0 to disable.")
    parser.add_argument("--min_avg_speed_kmh", type=float, default=1.0,
                        help="Drop entire vessel tracks whose average speed is below "
                             "this threshold in km/h (default: 1.0). "
                             "Removes moored/anchored vessels. Set 0 to disable.")

    # Bounding box — defaults crop to continental US + Gulf + Caribbean
    parser.add_argument("--lon_min", type=float, default=-130.0,
                        help="Min longitude (default: -130.0, drops Hawaii & Pacific)")
    parser.add_argument("--lon_max", type=float, default=-60.0,
                        help="Max longitude (default: -60.0)")
    parser.add_argument("--lat_min", type=float, default=10.0,
                        help="Min latitude (default: 10.0)")
    parser.add_argument("--lat_max", type=float, default=55.0,
                        help="Max latitude (default: 55.0)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()