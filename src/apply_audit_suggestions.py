#!/usr/bin/env python3
import pandas as pd, datetime as dt
from pathlib import Path

DATA = Path("data")
REVIEW = Path("out/review")

def load_csv(p, cols):
    if p.exists():
        df = pd.read_csv(p, dtype=str).fillna("")
    else:
        df = pd.DataFrame(columns=cols)
    # Ensure columns exist in the right order
    for c in cols:
        if c not in df.columns: df[c] = ""
    return df[cols]

def backup(p: Path):
    if not p.exists(): return
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    p.rename(p.with_suffix(p.suffix + f".bak-{ts}"))

# Aliases
aliases_ssot = DATA/"organization_aliases_master.csv"
aliases_suggest = REVIEW/"suggest_alias_additions.csv"
if aliases_suggest.exists():
    ssot = load_csv(aliases_ssot, ["Alias","To Org (id)"])
    sug  = pd.read_csv(aliases_suggest, dtype=str).fillna("")
    # normalize alias form back to canonical case as-is
    sug.columns = ["Alias","To Org (id)"]
    merged = pd.concat([ssot, sug], ignore_index=True)
    merged = merged[(merged["Alias"]!="") & (merged["To Org (id)"]!="")].drop_duplicates()
    backup(aliases_ssot)
    merged.to_csv(aliases_ssot, index=False)
    print(f"Updated {aliases_ssot} with {len(merged)-len(ssot)} new rows (after dedupe).")

# Domains
domains_ssot = DATA/"domain_to_org.csv"
domains_suggest = REVIEW/"suggest_domain_additions.csv"
if domains_suggest.exists():
    ssot = load_csv(domains_ssot, ["Domain","To Org (id)"])
    sug  = pd.read_csv(domains_suggest, dtype=str).fillna("")
    merged = pd.concat([ssot, sug], ignore_index=True)
    merged["Domain"] = merged["Domain"].str.lower().str.strip().str.replace(r"^www\.", "", regex=True)
    merged = merged[(merged["Domain"]!="") & (merged["To Org (id)"]!="")].drop_duplicates(subset=["Domain"])
    backup(domains_ssot)
    merged.to_csv(domains_ssot, index=False)
    print(f"Updated {domains_ssot} with {len(merged)-len(ssot)} effective new rows (after dedupe).")

# Parents
parents_ssot = DATA/"org_parents.csv"
parents_suggest = REVIEW/"suggest_parent_additions.csv"
if parents_suggest.exists():
    ssot = load_csv(parents_ssot, ["Org (id)","Parent Org (id)"])
    sug  = pd.read_csv(parents_suggest, dtype=str).fillna("")
    merged = pd.concat([ssot, sug], ignore_index=True)
    merged = merged[(merged["Org (id)"]!="") & (merged["Parent Org (id)"]!="")].drop_duplicates()
    backup(parents_ssot)
    merged.to_csv(parents_ssot, index=False)
    print(f"Updated {parents_ssot} with {len(merged)-len(ssot)} new rows (after dedupe).")
