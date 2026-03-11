#!/usr/bin/env python3
"""
Combine multiple raw AIS CSV files into one, then run process_ais_data.py.

Usage:
    python3 scripts/combine_days.py data/raw/AIS_2024_03_26.csv \
                                    data/raw/AIS_2024_03_27.csv \
                                    data/raw/AIS_2024_03_28.csv \
        --output data/raw/AIS_combined.csv

    # Then process the combined file:
    python3 scripts/process_ais_data.py data/raw/AIS_combined.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Combine multiple raw AIS CSV files.")
    parser.add_argument("inputs", nargs="+", help="Raw AIS CSV files to combine")
    parser.add_argument("-o", "--output", default="data/raw/AIS_combined.csv",
                        help="Output path for combined CSV (default: data/raw/AIS_combined.csv)")
    parser.add_argument("--id_col",   default="MMSI")
    parser.add_argument("--time_col", default="BaseDateTime")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for path in args.inputs:
        print(f"Loading {path} ...")
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
        print(f"  {len(df):,} rows, {df[args.id_col].nunique():,} vessels")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined):,} rows before dedup")

    # Deduplicate on (vessel, timestamp) — same ping may appear in adjacent day files
    combined = combined.drop_duplicates(subset=[args.id_col, args.time_col])
    print(f"Combined: {len(combined):,} rows after dedup")
    print(f"Unique vessels: {combined[args.id_col].nunique():,}")

    combined.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print("\nNext step:")
    print(f"  python3 scripts/process_ais_data.py {out_path}")


if __name__ == "__main__":
    main()
