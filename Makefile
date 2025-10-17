.PHONY: setup csv excel merge excel-all audit apply-audit orgkit clean

# 1) one-time env + deps
setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

# 2) fast CSV-only pass
csv:
	. .venv/bin/activate && python src/normalize_batch.py --csv-only

# 3) parametric Excel chunk: make excel START=0 LIMIT=25
excel:
	. .venv/bin/activate && python src/normalize_batch.py --start $(START) --limit $(LIMIT)

# 4) merge all chunk ZIPs into one deduped bundle
merge:
	. .venv/bin/activate && python src/merge_chunks.py

# 5) run all Excel chunks up to MAX (inclusive) in steps of LIMIT
#    e.g. make excel-all MAX=325 LIMIT=25
excel-all:
	@LIMIT=$${LIMIT:-25}; MAX=$${MAX:-325}; \
	for START in $$(seq 0 $$LIMIT $$MAX); do \
	  echo "==> Chunk $$START..$$(($$START + $$LIMIT - 1))"; \
	  . .venv/bin/activate && python src/normalize_batch.py --start $$START --limit $$LIMIT || exit 1; \
	done

# 6) audit new spreadsheets in data/ against SSOT
audit:
	. .venv/bin/activate && python src/audit_org_sources.py

# 7) apply audit suggestions (backs up SSOT files first)
apply-audit:
	. .venv/bin/activate && python src/apply_audit_suggestions.py

# 8) regenerate SSOT CSVs from the Org Kit workbook (optional rerun)
orgkit:
	. .venv/bin/activate && python src/extract_org_kit.py

# 9) wipe out/out/work extracted temp files (safe to keep normally)
clean:
	rm -rf out/work
