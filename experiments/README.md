# Experiments

This folder is the source of truth for reproducible runs and comparisons.

## Workflow

1. Pick an entry from `ablation_matrix.md`.
2. Run training/evaluation with fixed seed/config.
3. Add one row to `runs.csv`.
4. Add short notes in `findings.md` (wins, failures, surprises).

## Conventions

- `run_id`: unique tag, e.g. `act_rh4_s42_v1`
- `config_hash`: short fingerprint of config/checkpoint choices
- `status`: `planned`, `running`, `done`, `failed`
- `is_primary`: `yes` for the run used in README headline tables

## Minimum Repro Metadata Per Run

- policy/checkpoint name
- actions per inference (horizon mode)
- train data size
- eval seed
- eval episode count and max steps
- model parameter count

