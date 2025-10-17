# CHAM Normalizer
Scripts to normalize event/training data into clean CSVs for import.

## Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

## Usage
Put raw input files in `data/`. Then:
python src/normalize_batch.py --csv-only
# or Excel in chunks:
python src/normalize_batch.py --start 0 --limit 25
