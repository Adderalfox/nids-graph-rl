# Data

Source: University of Queensland NetFlow datasets
(https://staff.itee.uq.edu.au/marius/NIDS_datasets), version 3 (NF3).
All four share one 55-column schema (confirmed by the Phase 1a audit).

| Short name | Raw file (as downloaded)  | Role | Flows | Attack rate | Hourly blocks |
|------------|---------------------------|------|-------|-------------|---------------|
| unsw       | NF-UNSW-NB15-v3.csv       | source + target | 2.4M  | 5.4%  | 27  |
| cic        | NF-CICIDS2018-v3.csv      | source + target | 20.1M | 12.9% | 141 |
| ton        | NF-ToN-IoT-v3.csv         | source + target | 27.5M | 39.0% | 108 |
| bot        | NF-BoT-IoT-v3.csv         | target only     | 16.9M | 99.7% | 31  |

BoT-IoT is target-only: with 99.7% attacks it cannot teach a model what benign
looks like. On BoT we report detection rate plus false-positive rate on its 52k
benign flows.

Place raw CSVs in `data/raw/` (git-ignored).

## Audit (Phase 1a)

```bash
uv run python scripts/audit_data.py --file data/raw/<file> --name <short_name>
```

## Preparation (Phase 1b)

```bash
uv run python scripts/prepare_data.py --raw data/raw/NF-UNSW-NB15-v3.csv --name unsw
uv run python scripts/prepare_data.py --raw data/raw/NF-CICIDS2018-v3.csv --name cic
uv run python scripts/prepare_data.py --raw data/raw/NF-ToN-IoT-v3.csv --name ton
uv run python scripts/prepare_data.py --raw data/raw/NF-BoT-IoT-v3.csv --name bot
```

Output per dataset in `data/processed/<name>/`: `flows/block-*.parquet` (one file
per hour, sorted by start time), `hosts.parquet` (stable host id to IP, for
visualization only) and `prep_report.json`. Columns: metadata (`start_ms`,
`end_ms`, `src_id`, `dst_id`, `src_port`, `dst_port`, `label`, `attack`,
`attack_group`) followed by 55 features (transformed, not standardized).

## Splits, scalers and protocol (Phase 1c)

```bash
uv run python scripts/make_splits.py
```

Writes `data/processed/splits/<name>.json` (run lists, purged segments and
per-split statistics), `data/processed/scalers/<name>.json` (source datasets
only) and `data/processed/protocol.json` (every source-target pair with its
evaluation set and shared/unseen attack groups). Rows get their split from
`nidsrl.data.splits.assign_rows(start_ms, plan)`: 0 train, 1 val, 2 test,
-1 purged.

Method and rationale: `docs/adr/0004-evaluation-protocol.md` and
`docs/adr/0005-stratified-purged-splits.md` (chronological splits failed on the
real data because attack campaigns are scheduled in time).

Window sampling is not done here: window size and per-window edge caps are
graph-builder decisions (Phase 3).

## Coarse attack groups

benign, dos (all DoS/DDoS), recon (scanning, reconnaissance, analysis),
bruteforce (FTP/SSH/password/web brute force), web (injection, XSS, SQL
injection), exploit (exploits, fuzzers, generic, shellcode), malware (backdoor,
bot, ransomware, worms, infiltration), other (mitm, theft). See
`src/nidsrl/data/taxonomy.py`.

Observed overlap (flows): dos in all four; recon in unsw/ton/bot; bruteforce in
cic/ton; web in cic (920) and ton; malware in unsw/cic/ton; exploit only in unsw.
