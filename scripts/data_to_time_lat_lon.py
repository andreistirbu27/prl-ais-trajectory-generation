#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Trim AIS CSV to MMSI, BaseDateTime, LAT, LON and sort by MMSI + time."
    )
    parser.add_argument("input_csv", help="Path to input AIS CSV")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument(
        "--min_points",
        type=int,
        default=20,
        help="Minimum number of points per vessel track (default: 20)",
    )
    parser.add_argument(
        "--max_jump_deg",
        type=float,
        default=0.045,
        help=(
            "Maximum allowed spatial jump between consecutive points in degrees "
            "(default: 0.045 ≈ 5 km). Rows exceeding this are dropped."
        ),
    )
    args = parser.parse_args()

    inp = Path(args.input_csv)

    if args.output is None:
        out_dir = Path("data") / "processed"
        out_path = out_dir / f"{inp.stem}_processed.csv"
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ["MMSI", "BaseDateTime", "LAT", "LON"]
    df = pd.read_csv(inp, usecols=cols)
    print(f"Loaded:            {len(df):>10,} rows")

    # 1. Drop rows with any missing values
    df = df.dropna(subset=cols)
    print(f"After dropna:      {len(df):>10,} rows")

    # 2. Parse and validate timestamps
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce")
    df = df.dropna(subset=["BaseDateTime"])
    print(f"After ts parse:    {len(df):>10,} rows")

    # 3. Filter invalid coordinates
    valid_coords = df["LAT"].between(-90, 90) & df["LON"].between(-180, 180)
    df = df[valid_coords]
    print(f"After coord check: {len(df):>10,} rows")

    # 4. Remove duplicate (MMSI, timestamp) pairs — keep first occurrence
    df = df.drop_duplicates(subset=["MMSI", "BaseDateTime"])
    print(f"After dedup:       {len(df):>10,} rows")

    # 5. Sort by vessel then time (required for jump filter and training)
    df = df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

    # 6. Filter unrealistic spatial jumps (optional)
    if args.max_jump_deg is not None:
        dlon = df.groupby("MMSI")["LON"].diff()
        dlat = df.groupby("MMSI")["LAT"].diff()
        jump = np.sqrt(dlon**2 + dlat**2)
        # NaN = first point of each vessel track → always keep
        keep = (jump <= args.max_jump_deg) | jump.isna()
        df = df[keep].reset_index(drop=True)
        print(f"After jump filter: {len(df):>10,} rows  (max_jump={args.max_jump_deg}°)")

    # 7. Remove vessels with too few points
    counts = df.groupby("MMSI")["MMSI"].transform("count")
    df = df[counts >= args.min_points].reset_index(drop=True)
    print(f"After min_points:  {len(df):>10,} rows  (min={args.min_points})")

    df.to_csv(out_path, index=False)
    n_vessels = df["MMSI"].nunique()
    print(f"\nSaved: {out_path}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Vessels: {n_vessels:,}")


if __name__ == "__main__":
    main()
