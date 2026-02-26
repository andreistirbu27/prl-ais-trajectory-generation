#!/usr/bin/env python3
"""
Data Validation and Analysis Script for AIS Data

This script checks for common data quality issues that can cause
poor model performance:
- Invalid coordinates (out of bounds)
- Missing values
- Unrealistic jumps between consecutive points
- Temporal issues (unsorted, duplicates)
- Statistical outliers

Run this BEFORE training to understand your data!
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def validate_ais_data(csv_path: str, 
                      id_col: str = "MMSI",
                      time_col: str = "BaseDateTime",
                      lat_col: str = "LAT",
                      lon_col: str = "LON"):
    """Comprehensive validation of AIS CSV data."""
    
    print("="*70)
    print("AIS DATA VALIDATION REPORT")
    print("="*70)
    print(f"File: {csv_path}\n")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded CSV")
        print(f"  Total rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}\n")
    except Exception as e:
        print(f"✗ ERROR loading CSV: {e}")
        return
    
    # Check required columns
    print("-" * 70)
    print("1. CHECKING REQUIRED COLUMNS")
    print("-" * 70)
    required_cols = [id_col, time_col, lat_col, lon_col]
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"✗ MISSING COLUMNS: {missing}")
        return
    else:
        print(f"✓ All required columns present")
        for col in required_cols:
            print(f"  - {col}: {df[col].dtype}")
    print()
    
    # Check for NaN values
    print("-" * 70)
    print("2. CHECKING FOR MISSING VALUES")
    print("-" * 70)
    nan_counts = df[required_cols].isna().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        print(f"⚠ Found {total_nan:,} missing values:")
        for col, count in nan_counts.items():
            if count > 0:
                pct = 100 * count / len(df)
                print(f"  - {col}: {count:,} ({pct:.2f}%)")
    else:
        print(f"✓ No missing values found")
    print()
    
    # Validate coordinate ranges
    print("-" * 70)
    print("3. VALIDATING COORDINATE RANGES")
    print("-" * 70)
    lat_valid = df[lat_col].between(-90, 90)
    lon_valid = df[lon_col].between(-180, 180)
    
    n_invalid_lat = (~lat_valid).sum()
    n_invalid_lon = (~lon_valid).sum()
    
    if n_invalid_lat > 0:
        print(f"✗ Invalid latitudes: {n_invalid_lat:,}")
        print(f"  Range: [{df[lat_col].min():.4f}, {df[lat_col].max():.4f}]")
        print(f"  Examples of invalid: {df.loc[~lat_valid, lat_col].head().tolist()}")
    else:
        print(f"✓ All latitudes in valid range [-90, 90]")
        print(f"  Range: [{df[lat_col].min():.4f}, {df[lat_col].max():.4f}]")
    
    if n_invalid_lon > 0:
        print(f"✗ Invalid longitudes: {n_invalid_lon:,}")
        print(f"  Range: [{df[lon_col].min():.4f}, {df[lon_col].max():.4f}]")
        print(f"  Examples of invalid: {df.loc[~lon_valid, lon_col].head().tolist()}")
    else:
        print(f"✓ All longitudes in valid range [-180, 180]")
        print(f"  Range: [{df[lon_col].min():.4f}, {df[lon_col].max():.4f}]")
    print()
    
    # Clean data for further analysis
    df_clean = df[lat_valid & lon_valid].copy()
    df_clean = df_clean.dropna(subset=required_cols)
    print(f"After basic cleaning: {len(df_clean):,} rows ({100*len(df_clean)/len(df):.1f}%)\n")
    
    # Parse timestamps
    print("-" * 70)
    print("4. ANALYZING TIMESTAMPS")
    print("-" * 70)
    try:
        df_clean[time_col] = pd.to_datetime(df_clean[time_col], errors='coerce')
        n_invalid_time = df_clean[time_col].isna().sum()
        if n_invalid_time > 0:
            print(f"⚠ Could not parse {n_invalid_time:,} timestamps")
        else:
            print(f"✓ All timestamps parsed successfully")
        
        df_clean = df_clean.dropna(subset=[time_col])
        time_range = df_clean[time_col].max() - df_clean[time_col].min()
        print(f"  Time range: {df_clean[time_col].min()} to {df_clean[time_col].max()}")
        print(f"  Duration: {time_range}")
    except Exception as e:
        print(f"✗ Error parsing timestamps: {e}")
    print()
    
    # Check sorting
    print("-" * 70)
    print("5. CHECKING DATA SORTING")
    print("-" * 70)
    df_sorted = df_clean.sort_values([id_col, time_col])
    is_sorted = df_clean.equals(df_sorted)
    if is_sorted:
        print(f"✓ Data is already sorted by {id_col}, {time_col}")
    else:
        print(f"⚠ Data is NOT sorted by {id_col}, {time_col}")
        print(f"  → Run preprocessing script to sort data")
    print()
    
    # Analyze per-vessel statistics
    print("-" * 70)
    print("6. PER-VESSEL STATISTICS")
    print("-" * 70)
    vessel_counts = df_clean.groupby(id_col).size()
    print(f"Total vessels: {len(vessel_counts):,}")
    print(f"Points per vessel:")
    print(f"  Min:    {vessel_counts.min():,}")
    print(f"  Max:    {vessel_counts.max():,}")
    print(f"  Mean:   {vessel_counts.mean():.1f}")
    print(f"  Median: {vessel_counts.median():.1f}")
    
    n_short = (vessel_counts < 5).sum()
    if n_short > 0:
        print(f"\n⚠ {n_short:,} vessels have <5 points (will be excluded)")
    
    # Sample a few vessels for detailed analysis
    print(f"\nSample vessels:")
    for vid in vessel_counts.head(3).index:
        vcount = vessel_counts[vid]
        print(f"  {id_col}={vid}: {vcount} points")
    print()
    
    # Check for temporal duplicates
    print("-" * 70)
    print("7. CHECKING FOR DUPLICATE TIMESTAMPS")
    print("-" * 70)
    df_clean['_group'] = df_clean.groupby([id_col, time_col]).ngroup()
    duplicates = df_clean[df_clean.duplicated(subset=['_group'], keep=False)]
    n_dup_groups = duplicates['_group'].nunique()
    
    if n_dup_groups > 0:
        print(f"⚠ Found {len(duplicates):,} rows in {n_dup_groups:,} duplicate groups")
        print(f"  (Same vessel, same timestamp)")
        print(f"\nExample duplicate group:")
        example_group = duplicates['_group'].iloc[0]
        print(duplicates[duplicates['_group'] == example_group][[id_col, time_col, lat_col, lon_col]])
    else:
        print(f"✓ No duplicate timestamps found")
    df_clean = df_clean.drop('_group', axis=1)
    print()
    
    # Analyze jumps between consecutive points
    print("-" * 70)
    print("8. ANALYZING JUMPS BETWEEN CONSECUTIVE POINTS")
    print("-" * 70)
    
    jump_stats = []
    time_diff_stats = []
    
    for vid, group in df_clean.groupby(id_col):
        if len(group) < 2:
            continue
        
        group = group.sort_values(time_col)
        
        # Spatial jumps (degrees)
        lons = group[lon_col].values
        lats = group[lat_col].values
        dlon = np.diff(lons)
        dlat = np.diff(lats)
        jumps = np.sqrt(dlon**2 + dlat**2)
        jump_stats.extend(jumps)
        
        # Temporal gaps
        times = group[time_col].values
        time_diffs = np.diff(times.astype('datetime64[s]').astype(float))  # seconds
        time_diff_stats.extend(time_diffs)
    
    jump_stats = np.array(jump_stats)
    time_diff_stats = np.array(time_diff_stats)
    
    print(f"Spatial jumps (degrees):")
    print(f"  Mean:   {jump_stats.mean():.6f}° (~{jump_stats.mean()*111:.1f} km)")
    print(f"  Median: {np.median(jump_stats):.6f}° (~{np.median(jump_stats)*111:.1f} km)")
    print(f"  Max:    {jump_stats.max():.6f}° (~{jump_stats.max()*111:.1f} km)")
    print(f"  P95:    {np.percentile(jump_stats, 95):.6f}° (~{np.percentile(jump_stats, 95)*111:.1f} km)")
    print(f"  P99:    {np.percentile(jump_stats, 99):.6f}° (~{np.percentile(jump_stats, 99)*111:.1f} km)")
    
    # Flag unrealistic jumps (>1 degree = ~111km)
    n_large_jumps = (jump_stats > 1.0).sum()
    if n_large_jumps > 0:
        pct_large = 100 * n_large_jumps / len(jump_stats)
        print(f"\n⚠ {n_large_jumps:,} jumps > 1° (~111km) = {pct_large:.2f}% of all steps")
        print(f"  → These may be data errors or vessel teleportation!")
    
    print(f"\nTemporal gaps (seconds):")
    print(f"  Mean:   {time_diff_stats.mean():.1f}s (~{time_diff_stats.mean()/60:.1f} min)")
    print(f"  Median: {np.median(time_diff_stats):.1f}s (~{np.median(time_diff_stats)/60:.1f} min)")
    print(f"  Max:    {time_diff_stats.max():.1f}s (~{time_diff_stats.max()/3600:.1f} hours)")
    print(f"  P95:    {np.percentile(time_diff_stats, 95):.1f}s (~{np.percentile(time_diff_stats, 95)/60:.1f} min)")
    print()
    
    # Estimate vessel speeds
    print("-" * 70)
    print("9. ESTIMATED VESSEL SPEEDS")
    print("-" * 70)
    # Speed = distance / time
    # Convert degrees to km, seconds to hours
    distances_km = jump_stats * 111  # rough approximation
    times_hours = time_diff_stats / 3600
    
    # Filter out zero-time gaps to avoid division by zero
    valid_speeds = times_hours > 0
    speeds_kmh = distances_km[valid_speeds] / times_hours[valid_speeds]
    
    print(f"Speed between consecutive points (km/h):")
    print(f"  Mean:   {speeds_kmh.mean():.1f} km/h")
    print(f"  Median: {np.median(speeds_kmh):.1f} km/h")
    print(f"  Max:    {speeds_kmh.max():.1f} km/h")
    print(f"  P95:    {np.percentile(speeds_kmh, 95):.1f} km/h")
    
    # Typical vessel speeds: 10-30 knots = 18-55 km/h
    n_unrealistic = (speeds_kmh > 100).sum()  # >100 km/h is very fast for ships
    if n_unrealistic > 0:
        pct = 100 * n_unrealistic / len(speeds_kmh)
        print(f"\n⚠ {n_unrealistic:,} speeds > 100 km/h ({pct:.2f}%)")
        print(f"  → Possible data errors or very fast vessels (ferries?)")
    print()
    
    # Geographic distribution
    print("-" * 70)
    print("10. GEOGRAPHIC DISTRIBUTION")
    print("-" * 70)
    print(f"Latitude range:  [{df_clean[lat_col].min():.4f}, {df_clean[lat_col].max():.4f}]")
    print(f"Longitude range: [{df_clean[lon_col].min():.4f}, {df_clean[lon_col].max():.4f}]")
    
    lat_span = df_clean[lat_col].max() - df_clean[lat_col].min()
    lon_span = df_clean[lon_col].max() - df_clean[lon_col].min()
    print(f"Geographic extent: ~{lat_span*111:.0f} km × {lon_span*111:.0f} km")
    print()
    
    # Summary and recommendations
    print("="*70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*70)
    
    issues = []
    
    if total_nan > 0:
        issues.append(f"- Remove {total_nan:,} rows with missing values")
    
    if n_invalid_lat > 0 or n_invalid_lon > 0:
        issues.append(f"- Remove {n_invalid_lat + n_invalid_lon:,} rows with invalid coordinates")
    
    if not is_sorted:
        issues.append(f"- Sort data by {id_col} then {time_col}")
    
    if n_dup_groups > 0:
        issues.append(f"- Handle {len(duplicates):,} duplicate timestamps")
    
    if n_large_jumps > 0:
        issues.append(f"- Filter {n_large_jumps:,} unrealistic jumps (>111km)")
    
    if n_short > 0:
        issues.append(f"- Exclude {n_short:,} vessels with <5 points")
    
    if issues:
        print("⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\n→ Run the preprocessing script: data_to_time_lat_lon.py")
        print("→ Consider adding --max_jump_degrees flag to training script")
    else:
        print("✓ Data looks good! Ready for training.")
    
    print("\nKEY INSIGHTS:")
    print(f"- Average time between points: ~{time_diff_stats.mean()/60:.1f} minutes")
    print(f"- Average vessel speed: ~{speeds_kmh.mean():.1f} km/h")
    print(f"- With seq_len=20, you're predicting ~{20 * time_diff_stats.mean()/60:.1f} minutes ahead")
    
    predicted_time = 20 * time_diff_stats.mean() / 60
    predicted_dist = 20 * speeds_kmh.mean() * (time_diff_stats.mean() / 3600)
    print(f"- Expected travel distance in 20 steps: ~{predicted_dist:.1f} km")
    print(f"\n  If your model error is >20km, that's comparable to the distance traveled!")
    print(f"  → Consider reducing seq_len to 5-10 for better predictions")
    
    print("\n" + "="*70)
    

def plot_distributions(csv_path: str,
                       id_col: str = "MMSI",
                       time_col: str = "BaseDateTime", 
                       lat_col: str = "LAT",
                       lon_col: str = "LON",
                       output_dir: str = "data_validation"):
    """Create visualization plots."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[id_col, time_col, lat_col, lon_col])
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.sort_values([id_col, time_col])
    
    # Calculate jumps
    jumps = []
    for vid, group in df.groupby(id_col):
        if len(group) < 2:
            continue
        lons = group[lon_col].values
        lats = group[lat_col].values
        dlon = np.diff(lons)
        dlat = np.diff(lats)
        jump = np.sqrt(dlon**2 + dlat**2) * 111  # Convert to km
        jumps.extend(jump)
    
    jumps = np.array(jumps)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Geographic scatter
    ax = axes[0, 0]
    sample = df.sample(min(10000, len(df)))  # Sample for faster plotting
    ax.scatter(sample[lon_col], sample[lat_col], alpha=0.1, s=1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographic Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Jump distribution
    ax = axes[0, 1]
    ax.hist(jumps[jumps < 10], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Jump Distance (km)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Jumps (< 10km)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Vessel track lengths
    ax = axes[1, 0]
    lengths = df.groupby(id_col).size()
    ax.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Points per Vessel')
    ax.set_ylabel('Number of Vessels')
    ax.set_title('Track Length Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Large jumps
    ax = axes[1, 1]
    large_jumps = jumps[jumps > 1]
    if len(large_jumps) > 0:
        ax.hist(large_jumps, bins=50, alpha=0.7, edgecolor='black', color='red')
        ax.set_xlabel('Jump Distance (km)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Large Jumps (>{111:.0f}km) - Potential Errors')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No large jumps found!', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Large Jumps')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'data_quality.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plots to: {output_path}")
    

def main():
    parser = argparse.ArgumentParser(
        description="Validate AIS data quality"
    )
    parser.add_argument("csv", help="Path to AIS CSV file")
    parser.add_argument("--id_col", default="MMSI")
    parser.add_argument("--time_col", default="BaseDateTime")
    parser.add_argument("--lat_col", default="LAT")
    parser.add_argument("--lon_col", default="LON")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--plot_dir", default="data_validation",
                       help="Directory for plots")
    
    args = parser.parse_args()
    
    # Run validation
    validate_ais_data(
        args.csv,
        id_col=args.id_col,
        time_col=args.time_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col
    )
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        try:
            plot_distributions(
                args.csv,
                id_col=args.id_col,
                time_col=args.time_col,
                lat_col=args.lat_col,
                lon_col=args.lon_col,
                output_dir=args.plot_dir
            )
        except Exception as e:
            print(f"Error generating plots: {e}")


if __name__ == "__main__":
    main()
