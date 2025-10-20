#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

MERGED = Path("out/merged")
OUT = Path("out/review"); OUT.mkdir(exist_ok=True, parents=True)

def domain(email):
    email = str(email).lower()
    return email.split("@")[-1] if "@" in email else ""

# 1) Which org spellings show up the most?
orgs = pd.read_csv(MERGED/"unmatched_orgs.csv", dtype=str).fillna("")
orgs = orgs["org_name_normalized"].value_counts().reset_index()
orgs.columns = ["org_name_normalized","count"]
orgs.to_csv(OUT/"unmatched_orgs_ranked.csv", index=False)

# 2) Which email domains are most common among unmatched contacts?
uc = pd.read_csv(MERGED/"unmatched_contacts.csv", dtype=str).fillna("")
uc["domain"] = uc["email"].map(domain)
dom = uc[uc["domain"]!=""]["domain"].value_counts().reset_index()
dom.columns = ["domain","count"]
dom.to_csv(OUT/"unmatched_contact_domains_ranked.csv", index=False)

print("Wrote:")
print(" - out/review/unmatched_orgs_ranked.csv")
print(" - out/review/unmatched_contact_domains_ranked.csv")
