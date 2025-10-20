#!/usr/bin/env python3
"""
Extract canonical orgs, aliases, parents, and domains from the Org Normalization Kit.
Outputs:
  data/canonical_orgs_master_final.csv
  data/organization_aliases_master.csv
  data/org_parents.csv
  data/domain_to_org.csv  (if ContactsOrgs-like sheet exists)
"""

import pandas as pd
from pathlib import Path

DATA = Path("data")
SRC  = DATA / "Org_Normalization_Kit_Repaired_Recomputed.xlsx"

FREE = {
    "gmail.com","yahoo.com","outlook.com","hotmail.com","aol.com","icloud.com",
    "comcast.net","att.net","msn.com","live.com","proton.me","protonmail.com",
    "ymail.com","me.com","mail.com","gmx.com","cox.net","sbcglobal.net"
}

def coerce(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().astype(str).fillna("").applymap(lambda x: x.strip())

def find_sheet(xl: pd.ExcelFile, candidates: list[str]) -> str | None:
    lower = {s.lower(): s for s in xl.sheet_names}
    for want in candidates:
        if want in lower:           # exact lower match
            return lower[want]
    # loose contains
    for s in xl.sheet_names:
        sl = s.lower()
        if any(w in sl for w in candidates):
            return s
    return None

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Missing {SRC}. Put the Org Kit in data/")

    xl = pd.ExcelFile(SRC)

    # --- Winners (canonical + parents)
    winners_sheet = find_sheet(xl, ["winners", "winner"])
    if not winners_sheet:
        raise RuntimeError("Could not find a 'Winners' sheet in the Org Kit.")

    winners = coerce(pd.read_excel(SRC, sheet_name=winners_sheet, dtype=str))

    # Column aliases
    col_winner_id = next((c for c in winners.columns if c.lower() in {"winnerorgid","org_id","orgid","winner org id"}), None)
    col_canon     = next((c for c in winners.columns if c.lower() in {"canonicalname","orgname","name (canonical)"}), None)
    col_iswinner  = next((c for c in winners.columns if c.lower() in {"iswinner","is_winner"}), None)
    col_parent_id = next((c for c in winners.columns if c.lower() in {"parentorgid","parent org (id)","parent_org_id"}), None)

    if not (col_winner_id and col_canon):
        raise RuntimeError("Winners sheet missing WinnerOrgId / CanonicalName columns.")

    w = winners[[col_winner_id, col_canon] + ([col_iswinner] if col_iswinner else [])].copy()
    w.columns = ["org_id", "Name (canonical)"] + (["IsWinner"] if col_iswinner else [])

    if "IsWinner" in w.columns:
        mask = w["IsWinner"].str.lower().isin({"1","true","yes","y"})
        w = w[mask]

    # drop blanks safely (PANDAS TIP: wrap each comparison in parentheses)
    mask_keep = (w["org_id"].ne("")) & (w["Name (canonical)"].ne(""))
    canon_out = w.loc[mask_keep, ["org_id","Name (canonical)"]].drop_duplicates()
    (DATA / "canonical_orgs_master_final.csv").write_text(canon_out.to_csv(index=False))
    print(f"Wrote data/canonical_orgs_master_final.csv with {len(canon_out)} rows.")

    # --- Parents
    if col_parent_id:
        p = winners[[col_winner_id, col_parent_id]].copy()
        p.columns = ["Org (id)", "Parent Org (id)"]
        p = coerce(p)
        p = p[(p["Org (id)"].ne("")) & (p["Parent Org (id)"].ne(""))].drop_duplicates()
        (DATA / "org_parents.csv").write_text(p.to_csv(index=False))
        print(f"Wrote data/org_parents.csv with {len(p)} rows.")
    else:
        print("No ParentOrgId column found; skipped parent export.")

    # --- Aliases
    aliases_sheet = find_sheet(xl, ["aliases","alias"])
    if not aliases_sheet:
        print("No 'Aliases' sheet found; skipping alias export.")
    else:
        aliases = coerce(pd.read_excel(SRC, sheet_name=aliases_sheet, dtype=str))
        a_col = next((c for c in aliases.columns if c.lower() in {"alias","aliasname"}), None)
        t_col = next((c for c in aliases.columns if c.lower() in {"winnerorgid","to org (id)","to_org_id"}), None)
        if a_col and t_col:
            a = aliases[[a_col, t_col]].copy()
            a.columns = ["Alias","To Org (id)"]
            a = a[(a["Alias"].ne("")) & (a["To Org (id)"].ne(""))].drop_duplicates()
            (DATA / "organization_aliases_master.csv").write_text(a.to_csv(index=False))
            print(f"Wrote data/organization_aliases_master.csv with {len(a)} rows.")
        else:
            print("Aliases sheet missing Alias/WinnerOrgId; skipped alias export.")

    # --- Domains (optional)
    dom_sheet = find_sheet(xl, ["contactsorgs","contacts orgs","domains","contact"])
    if dom_sheet:
        domdf = coerce(pd.read_excel(SRC, sheet_name=dom_sheet, dtype=str))
        d_norm = next((c for c in domdf.columns if c.lower() in {"domainprimarynorm","domain"}), None)
        d_raw  = next((c for c in domdf.columns if c.lower() in {"domainprimary","emaildomain"}), None)
        t_col  = next((c for c in domdf.columns if c.lower() in {"winnerorgid","orgid","to org (id)"}), None)
        if t_col and (d_norm or d_raw):
            dom = domdf[d_norm] if d_norm else domdf[d_raw]
            dom = dom.str.lower().str.strip().str.replace(r"^www\.", "", regex=True)
            tgt = domdf[t_col]
            dmap = pd.DataFrame({"Domain": dom, "To Org (id)": tgt})
            dmap = dmap[(dmap["Domain"].ne("")) & (dmap["To Org (id)"].ne(""))]
            dmap = dmap[~dmap["Domain"].isin(FREE)]
            dmap = dmap.drop_duplicates(subset=["Domain"]).sort_values("Domain")
            (DATA / "domain_to_org.csv").write_text(dmap.to_csv(index=False))
            print(f"Wrote data/domain_to_org.csv with {len(dmap)} rows (from Org Kit).")
        else:
            print("ContactsOrgs-like sheet lacks domain/target columns; skipped domain export.")
    else:
        print("No ContactsOrgs/domains sheet found; domain export already handled elsewhere if present.")

if __name__ == "__main__":
    main()
