.PHONY: setup csv excel0 excel1

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

csv:
	. .venv/bin/activate && python src/normalize_batch.py --csv-only

excel0:
	. .venv/bin/activate && python src/normalize_batch.py --start 0 --limit 25

excel1:
	. .venv/bin/activate && python src/normalize_batch.py --start 25 --limit 25
