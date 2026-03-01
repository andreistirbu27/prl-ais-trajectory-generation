# PRL: AIS Trajectory Generation

Transformer-based next-step prediction and synthetic maritime trajectory generation from AIS tracklines.

## Structure
- src/        core code
- scripts/    CLI scripts
- notebooks/  exploration
- configs/    experiment configs
- docs/       notes
- results/    figures/metrics
- data/       datasets (not tracked)
- outputs/    checkpoints/preds (not tracked)

ais_prepare.py — AIS Data Preprocessing Pipeline

Takes a raw AIS CSV, applies a sequence of cleaning filters, writes the
processed output, and prints a full audit log showing exactly how many rows
each filter removed and why.

Usage:
    python ais_prepare.py data/raw/ais.csv
    python ais_prepare.py data/raw/ais.csv -o data/processed/ais_clean.csv
    python ais_prepare.py data/raw/ais.csv --max_jump_km 10 --max_gap_min 30
    python ais_prepare.py data/raw/ais.csv --min_avg_speed_kmh 1.0




train_ais_transformer.py — AIS Vessel Trajectory Prediction

Transformer encoder that predicts the next position for each timestep
in a sliding window sequence (seq2seq, shifted by 1).

Input features per timestep  : [lon, lat, dt]         (3-dim, default)
                               + [dlon, dlat]          (5-dim with --use_velocity)
Target per timestep          : [lon, lat]              (2-dim, shifted +1)

Key design decisions:
  - Bidirectional attention (no causal mask): consistent with how the model
    is used at inference time (whole window in, whole shifted window out).
  - dt (time-delta) as an explicit input feature: your data has variable gaps
    (mean 102s, p95 181s) so the model needs to know how much time passed.
  - Cosine LR schedule with linear warmup.
  - Full checkpoint saves args + scaler + metrics so the model is self-contained.

Usage:
    python train_ais_transformer.py --csv data/processed/ais.csv
    python train_ais_transformer.py --csv data/processed/ais.csv \\
        --seq_len 20 --nhid 256 --nlayers 4 --epochs 30 --use_velocity