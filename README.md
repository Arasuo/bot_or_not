# Bot or Not Detector

A Python script for the Bot-or-Not challenge that detects likely bot accounts from user + post data.

- Script: `detect_bots.py`
- Output format: one user ID per line 
- Scoring-aware evaluation: `+4` (true bot), `-1` (missed bot), `-2` (human flagged as bot)

## Features

- Adaptive detection mode (default) using per-dataset quantile normalization.
- Legacy mode for compatibility and baseline comparison.
- Confidence-tiered decisions (`high` / `medium`) with conservative guardrails.
- Repetition signals:
  - Exact duplicate content.
  - Near-duplicate hashing (URLs/mentions/hashtags/punctuation normalized).
- Temporal behavior signals (posting regularity + burstiness).
- Optional parameter tuning across labeled datasets.
- Optional bootstrap robustness check.
- Explainable outputs (`--explain-top`) and diagnostics CSV export.

## Requirements

- Python 3.9+
- Standard library only (no external dependencies)

## Input Format

The JSON input must include top-level lists:

- `users`: user objects with fields like `id`, `username`, `description`, `location`, `tweet_count`, `z_score`
- `posts`: post objects with fields like `author_id`, `text`, `created_at`

Optional ground truth for evaluation:

- `dataset.bots.txt`: one bot user ID per line

## Quick Start

Run adaptive mode and write challenge submission file:

```bash
python3 detect_bots.py \
  --mode adaptive \
  --input-json 'dataset.posts&users.json' \
  --output-txt MyTeam.detections.en.txt
```

## Evaluate Against Ground Truth

```bash
python3 detect_bots.py \
  --mode adaptive \
  --input-json 'dataset.posts&users.json' \
  --bots-txt dataset.bots.txt \
  --output-txt MyTeam.detections.en.txt
```

This prints score, TP, FN, FP, precision, recall.

## Tune on Labeled Datasets

```bash
python3 detect_bots.py \
  --mode adaptive \
  --input-json 'dataset.posts&users.30.json' \
  --bots-txt dataset.bots.30.txt \
  --tune-dataset 'dataset.posts&users.30.json' dataset.bots.30.txt \
  --tune-dataset 'dataset.posts&users.32.json' dataset.bots.32.txt \
  --output-txt MyTeam.detections.en.txt
```

## Useful Options

- `--mode {adaptive,legacy}`: detector type (`adaptive` default)
- `--no-tune`: skip tuning and use defaults
- `--min-posts-for-timing 8`: minimum posts to trust timing features
- `--min-posts-for-repetition 6`: minimum posts to trust repetition features
- `--calibration-objective {robust,mean}`: tuning target (`robust` default)
- `--bootstrap-samples N`: run bootstrap robustness evaluation
- `--bootstrap-frac 0.8`: fraction of users per bootstrap sample
- `--explain-top N`: print top flagged users with reason codes
- `--diagnostics-csv diagnostics.csv`: write detailed per-user diagnostics

## Output

Submission file example:

```text
user-id-1
user-id-2
user-id-3
```

No header, one user ID per line.

## Reason Codes (Adaptive Mode)

Examples shown by `--explain-top`:

- `ACTIVITY_HIGH`, `ZSCORE_HIGH`
- `REPETITION_EXACT`, `REPETITION_RATIO`, `REPETITION_NEAR`
- `TIMING_REGULAR`, `BURSTY_POSTING`
- `HASHTAG_HEAVY`, `MENTION_HEAVY`
- `PROFILE_WEAK`, `BOT_KEYWORD`
- `INSUFFICIENT_SUPPORT`

## Notes

- Adaptive mode is designed to generalize across datasets by using dataset-relative thresholds.
