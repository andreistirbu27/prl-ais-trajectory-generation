# PRL: AIS Trajectory Generation

Transformer-based next-step prediction and synthetic maritime trajectory generation from AIS tracklines.

## Structure
- `scripts/`   CLI scripts (preprocessing + training)
- `docs/`      supervisor meeting notes and reference papers
- `data/`      datasets (not tracked)
- `runs/`      training outputs / checkpoints (not tracked)

## Data Pipeline

```
Raw AIS CSVs (marinecadastre.gov)
    ↓  scripts/combine_days.py       # merge multi-day files, deduplicate
    ↓  scripts/process_ais_data.py  # filter by vessel type, bbox, quality
    →  data/processed/<name>_processed.csv
```

### Combine multiple raw days
```bash
python3 scripts/combine_days.py data/raw/AIS_2024_03_26.csv \
    data/raw/AIS_2024_03_27.csv data/raw/AIS_2024_03_28.csv \
    --output data/raw/AIS_combined.csv
```

### Preprocess (audit log printed to stdout)
```bash
python3 scripts/process_ais_data.py data/raw/AIS_combined.csv \
    -o data/processed/AIS_combined_processed.csv
```
Keeps vessel types 60–89 (passenger/cargo/tanker), continental US bbox,
min 50 points per track, max jump 5 km, max gap 60 min, border truncation filter.

## Training

```bash
python3 scripts/train_ais_transformer.py \
    --csv data/processed/AIS_combined_processed.csv \
    --epochs 20 --val_frac 0.15

# With larger model + velocity features:
python3 scripts/train_ais_transformer.py \
    --csv data/processed/AIS_combined_processed.csv \
    --seq_len 20 --d_model 256 --num_layers 4 --epochs 30 --use_velocity
```

### Key design decisions

- **Displacement target** `[dlon_norm, dlat_norm]`, not absolute position.
  The model learns "how far/which direction" — a small-variance target regardless of where on the coast the vessel is.
  Absolute next position is recovered at eval: `pred_pos = input_pos + disp_scaler.inverse(pred)`.
- **Three separate scalers**: position (lon/lat), log(dt), and displacement.
  Each has its own mean/std, preventing any dimension from dominating the MSE loss.
- **Causal mask** (default): position t cannot attend to t+1, t+2, … — no future leakage.
- **Temporal gap mask**: positions with dt > `--max_gap_sec` (default 600 s) are masked so the model cannot attend across large data gaps.
- **Cosine LR + warmup**: 5% linear warmup then cosine decay.
- **Constant-velocity baseline** is printed before training. Beat this to show the transformer adds value.

### Input / target dimensions

| Arg              | Features per timestep                     | Dim |
|------------------|-------------------------------------------|-----|
| (default)        | `[lon_norm, lat_norm, log_dt_norm]`       | 3   |
| `--use_velocity` | above + `[dlon_norm, dlat_norm]`          | 5   |

Target per timestep: `[dlon_norm, dlat_norm]` (normalised displacement to next position)
