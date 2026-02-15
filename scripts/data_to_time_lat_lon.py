#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Trim AIS CSV to MMSI, BaseDateTime, LAT, LON and sort by MMSI + time."
    )
    parser.add_argument("input_csv", help="Path to input AIS CSV")
    parser.add_argument("-o", "--output", default=None)
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

    df = df.dropna()
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce")
    df = df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | rows: {len(df)}")

if __name__ == "__main__":
    main()
