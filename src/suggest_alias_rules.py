#!/usr/bin/env python3
import pandas as pd, re
from pathlib import Path

DATA = Path("data")
MERGED = Path("out/merged")
OUT = Path("out/review"); OUT.mkdir(parents=True, exist_ok=True)

# Load SSOT
canon = pd.read_csv(DATA/"canonical_orgs_master_final.csv", dtype=str).fillna("")
canon["nk"] = canon["Name (canonical)"].str.lower().str.replace(r"[^a-z0-9 ]","", regex=True).str.replace(r"\s+"," ", regex=True).str.strip()
id_by_nk = dict(zip(canon["nk"], canon["org_id"]))

# US state abbreviations
ST = set("al ak az ar ca co ct dc de fl ga hi ia id il in ks ky la ma md me mi mn mo ms mt nc nd ne nh nj nm nv ny oh ok or pa pr ri sc sd tn tx ut va vt wa wi wv wy".split())

def normalize_name(s: str) -> str:
    s = str(s or "").strip().lower()
    # remove DBA… segments
    s = re.sub(r"\bdba\b.*$", "", s)
    # drop " - 6-digit" trailing codes
    s = re.sub(r"\s*-\s*\d{4,7}$", "", s)
    # drop trailing state abbrev
    s = re.sub(r"\s+([a-z]{2})$", lambda m: "" if m.group(1) in ST else m.group(0), s)
    # collapse runs of spaces and punctuation
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # nuke obvious address tails (zip)
    s = re.sub(r"\b\d{5}(?:-\d{4})?$", "", s).strip()
    return s

def looks_like_address(s: str) -> bool:
    s = s.lower()
    # #digit + street OR has city/state/zip pattern
    return bool(re.search(r"\d{2,} [a-z].*(?: st| ave| rd| blvd| dr)\b", s)) or bool(re.search(r"\b[a-z]{2}\s*\d{5}(-\d{4})?\b", s))

unmatched = pd.read_csv(MERGED/"unmatched_orgs.csv", dtype=str).fillna("")
unmatched["orig"] = unmatched["org_name_normalized"].astype(str)
unmatched["nk"] = unmatched["orig"].str.lower().str.replace(r"[^a-z0-9 ]"," ", regex=True).str.replace(r"\s+"," ", regex=True).str.strip()
unmatched["rule_norm"] = unmatched["orig"].map(normalize_name)

# filter junk rows that are clearly addresses or dashes
cand = unmatched[~unmatched["orig"].str.strip().isin({"", "-", "n", "na", "n/a"})]
cand = cand[~cand["orig"].map(looks_like_address)]

# exact hits first (after rule_norm)
hits = cand[cand["rule_norm"].isin(id_by_nk.keys())].copy()
hits["To Org (id)"] = hits["rule_norm"].map(id_by_nk)
hits = hits[["orig","To Org (id)"]].drop_duplicates()
hits.columns = ["Alias","To Org (id)"]

# write suggestions + leftovers
hits.to_csv(OUT/"suggest_alias_additions_rules.csv", index=False)

left = cand[~cand["rule_norm"].isin(id_by_nk.keys())][["orig","rule_norm"]].drop_duplicates()
left.to_csv(OUT/"alias_leftovers_after_rules.csv", index=False)

print("Suggested alias rows:", len(hits))
print("Leftovers after rules:", len(left))
print("Wrote:", OUT/"suggest_alias_additions_rules.csv", "and", OUT/"alias_leftovers_after_rules.csv")
