#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

DATA = Path("data")
OUT = Path("out/review"); OUT.mkdir(parents=True, exist_ok=True)

# SSOT files (leave in data/)
CANON  = DATA / "canonical_orgs_master_final.csv"
ALIASES = DATA / "organization_aliases_master.csv"
DOMAINS = DATA / "domain_to_org.csv"
PARENTS = DATA / "org_parents.csv"

FREE_DOMAINS = {
    "gmail.com","yahoo.com","outlook.com","hotmail.com","aol.com","icloud.com",
    "comcast.net","att.net","msn.com","live.com","proton.me","protonmail.com",
    "ymail.com","me.com","mail.com","gmx.com","cox.net","sbcglobal.net"
}

def load_csv(path, **kw):
    if not path.exists(): return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, **kw).fillna("")
    except Exception:
        return pd.read_csv(path, dtype=str, encoding="latin1", **kw).fillna("")

def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower()==".csv":
        return load_csv(path)
    xl = pd.ExcelFile(path)
    frames = []
    for s in xl.sheet_names:
        try:
            df = xl.parse(s, dtype=str).fillna("")
            df["__sheet__"] = s
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def nrm(s): return str(s or "").strip().lower()

# --- Load SSOT ---
canon   = load_csv(CANON)
aliases = load_csv(ALIASES)
domains = load_csv(DOMAINS)
parents = load_csv(PARENTS)

alias_pairs  = {(nrm(a), str(t).strip()) for a,t in zip(aliases.get("Alias",[]), aliases.get("To Org (id)",[])) if str(a).strip() and str(t).strip()}
domain_pairs = {(nrm(d), str(t).strip()) for d,t in zip(domains.get("Domain",[]), domains.get("To Org (id)",[])) if str(d).strip() and str(t).strip()}
parent_pairs = {(str(c).strip(), str(p).strip()) for c,p in zip(parents.get("Org (id)",[]), parents.get("Parent Org (id)",[])) if str(c).strip() and str(p).strip()}

suggest_alias, suggest_domain, suggest_parent = set(), set(), set()
conflicts = {"alias": [], "domain": [], "parent": []}

ignored = {
    "canonical_orgs_master_final.csv",
    "organization_aliases_master.csv",
    "domain_to_org.csv",
    "org_parents.csv",
}

for path in sorted(DATA.glob("*")):
    if not path.is_file(): continue
    if path.name in ignored: continue
    if path.suffix.lower() not in {".csv",".xlsx",".xls"}: continue

    df = load_any(path)
    if df.empty: continue

    cols = {c.lower().strip(): c for c in df.columns}

    # Aliases
    a_col = next((cols[c] for c in cols if c in {"alias","aliasname"}), None)
    t_col = next((cols[c] for c in cols if c in {"to org (id)","winnerorgid","to_org_id","winner"}), None)
    if a_col and t_col:
        for a,t in zip(df[a_col], df[t_col]):
            a_norm, tgt = nrm(a), str(t).strip()
            if not a_norm or not tgt: continue
            pair = (a_norm, tgt)
            if pair in alias_pairs: continue
            existing = [x for x in alias_pairs if x[0]==a_norm]
            if existing and existing[0][1] != tgt:
                conflicts["alias"].append((path.name, a, tgt, "vs", existing[0][1]))
            else:
                suggest_alias.add(pair)

    # Domains
    d_col = next((cols[c] for c in cols if c in {"domainprimarynorm","emaildomain","domain"}), None)
    if d_col and t_col:
        for d,t in zip(df[d_col], df[t_col]):
            dom, tgt = nrm(d), str(t).strip()
            if not dom or not tgt or dom in FREE_DOMAINS: continue
            pair = (dom, tgt)
            if pair in domain_pairs: continue
            existing = [x for x in domain_pairs if x[0]==dom]
            if existing and existing[0][1] != tgt:
                conflicts["domain"].append((path.name, dom, tgt, "vs", existing[0][1]))
            else:
                suggest_domain.add(pair)

    # Parents
    c_col = next((cols[c] for c in cols if c in {"org (id)","orgid"}), None)
    p_col = next((cols[c] for c in cols if c in {"parent org (id)","parentorgid"}), None)
    if c_col and p_col:
        for cid,pid in zip(df[c_col], df[p_col]):
            cid, pid = str(cid).strip(), str(pid).strip()
            if not cid or not pid: continue
            pair = (cid, pid)
            if pair in parent_pairs: continue
            existing = [x for x in parent_pairs if x[0]==cid]
            if existing and existing[0][1] != pid:
                conflicts["parent"].append((path.name, cid, pid, "vs", existing[0][1]))
            else:
                suggest_parent.add(pair)

# Write outputs
pd.DataFrame(sorted(suggest_alias),  columns=["Alias_norm","To Org (id)"]).to_csv(OUT/"suggest_alias_additions.csv", index=False)
pd.DataFrame(sorted(suggest_domain), columns=["Domain","To Org (id)"]).to_csv(OUT/"suggest_domain_additions.csv", index=False)
pd.DataFrame(sorted(suggest_parent), columns=["Org (id)","Parent Org (id)"]).to_csv(OUT/"suggest_parent_additions.csv", index=False)

rows=[]
for k in ("alias","domain","parent"):
    for rec in conflicts[k]:
        rows.append((k,)+rec)
pd.DataFrame(rows, columns=["type","file","key","proposed","_","existing"]).to_csv(OUT/"conflicts_report.csv", index=False)

print("Audit complete. See out/review/:")
print("  - suggest_alias_additions.csv")
print("  - suggest_domain_additions.csv")
print("  - suggest_parent_additions.csv")
print("  - conflicts_report.csv")
