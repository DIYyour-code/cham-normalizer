#!/usr/bin/env python3
import pandas as pd, numpy as np, zipfile, re, hashlib, argparse
from pathlib import Path

def norm_text(s):
    s = str(s or "").strip()
    return re.sub(r"\s+", " ", s)

def norm_key(s):
    s = str(s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 @\-\_\.]", "", s)
    return s.strip()

def hash_id(*parts):
    h = hashlib.sha256("|".join([str(p or "").lower().strip() for p in parts]).encode("utf-8")).hexdigest()
    return h[:16]

def col(df,*keys):
    m = {str(k).lower():k for k in df.columns}
    for k in keys:
        if k in m: return df[m[k]]
    for c in m:
        if any(k in c for k in keys): return df[m[c]]
    return pd.Series([""]*len(df))

def classify(cols):
    L = [str(c).lower() for c in cols]
    score = {"attendance":0,"events":0,"certificates":0,"payments":0,"catalog":0}
    for c in L:
        if any(k in c for k in ["attend","roster","registrant","participant","seat"]): score["attendance"]+=1
        if any(k in c for k in ["event","start","end","location","host","venue"]): score["events"]+=1
        if any(k in c for k in ["cert","complete","graduat"]): score["certificates"]+=1
        if any(k in c for k in ["order","payment","amount","txn","invoice"]): score["payments"]+=1
        if any(k in c for k in ["course","training","catalog","code","ce hours","ceu"]): score["catalog"]+=1
    return max(score, key=score.get)

def read_csv_fast(p: Path):
    try:    return pd.read_csv(p, dtype=str).fillna("")
    except: return pd.read_csv(p, dtype=str, encoding="latin1").fillna("")

def read_excel_best_sheet(p: Path):
    xl = pd.ExcelFile(p)
    best = max(xl.sheet_names, key=lambda s: xl.parse(s, nrows=3).shape[1])
    return xl.parse(best, dtype=str).fillna("")

def unzip_select(zpath: Path, exts, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    with zipfile.ZipFile(zpath, "r") as z:
        for n in z.namelist():
            if n.endswith("/"): continue
            if not any(n.lower().endswith(ext) for ext in exts): continue
            dst = outdir / Path(n).name
            if not dst.exists():
                with z.open(n) as src, open(dst, "wb") as f: f.write(src.read())
            paths.append(dst)
    return paths

def build_org_lookup(canon_path: Path, alias_path: Path|None):
    canon = pd.read_csv(canon_path, dtype=str).fillna("")
    lookup = {norm_key(r["Name (canonical)"]): str(r["org_id"]).strip() for _, r in canon.iterrows()}
    if alias_path and alias_path.exists():
        alias_df = pd.read_csv(alias_path, dtype=str).fillna("")
        a_col = next((c for c in alias_df.columns if c.lower().strip() in {"alias","aliasname"}), None)
        t_col = next((c for c in alias_df.columns if c.lower().strip() in {"to org (id)","winnerorgid","to_org_id"}), None)
        if a_col and t_col:
            for _, r in alias_df.iterrows():
                a, t = r.get(a_col), r.get(t_col)
                if a and t:
                    lookup[norm_key(a)] = str(t).strip()
    return lookup

def build_contact_email_set(contacts_xlsx: Path):
    xl = pd.ExcelFile(contacts_xlsx)
    pick = next((s for s in xl.sheet_names if "email" in " ".join(map(str, xl.parse(s, nrows=1).columns)).lower()), xl.sheet_names[0])
    df = xl.parse(pick, dtype=str).fillna("")
    email_col = next((c for c in df.columns if "email" in c.lower()), df.columns[0])
    return {e for e in df[email_col].astype(str).str.lower().str.strip() if e}

def main():
    ap = argparse.ArgumentParser(description="CHAM Batch A normalizer (local, chunked) with alias/domain/parent mapping.")
    ap.add_argument("--data-dir", default="data", help="Folder with zips and master files")
    ap.add_argument("--out-dir",  default="out",  help="Output folder")
    ap.add_argument("--start", type=int, default=0, help="Start index for Excel files")
    ap.add_argument("--limit", type=int, default=25, help="Excel files per run")
    ap.add_argument("--csv-only", action="store_true", help="Process only CSVs (fast pass)")
    args = ap.parse_args()

    base = Path(".").resolve()
    data_dir = (base / args.data_dir)
    out_dir  = (base / args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"; work_dir.mkdir(exist_ok=True)

    canon_path    = data_dir / "canonical_orgs_master_final.csv"
    contacts_xlsx = data_dir / "contacts-master-union-925-postor.xlsx"
    alias_path    = data_dir / "organization_aliases_master.csv"
    domain_path   = data_dir / "domain_to_org.csv"
    parents_path  = data_dir / "org_parents.csv"

    org_lookup     = build_org_lookup(canon_path, alias_path if alias_path.exists() else None)
    contact_emails = build_contact_email_set(contacts_xlsx)

    domain_map = {}
    if domain_path.exists():
        ddf = pd.read_csv(domain_path, dtype=str).fillna("")
        d_col = next((c for c in ddf.columns if c.lower() in {"domain","emaildomain","domainprimarynorm"}), None)
        t_col = next((c for c in ddf.columns if c.lower() in {"to org (id)","winnerorgid","to_org_id"}), None)
        if d_col and t_col:
            # normalize domains (lowercase, strip, drop leading www.)
            d_ser = ddf[d_col].astype(str).str.lower().str.strip().str.replace(r"^www\.", "", regex=True)
            domain_map = {d: str(t).strip() for d, t in zip(d_ser, ddf[t_col]) if d and str(t).strip()}

    parent_map = {}
    if parents_path.exists():
        pdf = pd.read_csv(parents_path, dtype=str).fillna("")
        c_col = next((c for c in pdf.columns if c.lower() in {"org (id)","orgid"}), None)
        p_col = next((c for c in pdf.columns if c.lower() in {"parent org (id)","parentorgid"}), None)
        if c_col and p_col:
            for _, r in pdf.iterrows():
                cid, pid = str(r.get(c_col,"")).strip(), str(r.get(p_col,"")).strip()
                if cid and pid: parent_map[cid] = pid

    # unzip sources
    zip1 = data_dir / "events data exports.zip"
    zip2 = data_dir / "New Training Data 2.zip"
    csv_files, xls_files = [], []
    for z in [zip1, zip2]:
        if z.exists():
            csv_files += unzip_select(z, [".csv"], work_dir)
            xls_files += unzip_select(z, [".xlsx",".xls"], work_dir)

    files = csv_files if args.csv_only else (csv_files + xls_files[args.start: args.start + args.limit])

    # collectors
    catalog_rows, events_rows, attendance_rows, cert_rows, pay_rows = [], [], [], [], []
    unmatched_orgs, unmatched_contacts = set(), set()

    # helper
    def pick_domain(e):
        e = str(e).lower().strip()
        dom = e.split("@")[-1] if "@" in e else ""
        if dom.startswith("www."): dom = dom[4:]
        return dom

    # process files
    for p in files:
        try:
            if args.csv_only:
                df = read_csv_fast(p)
            else:
                df = read_excel_best_sheet(p) if p.suffix.lower() in (".xlsx",".xls") else read_csv_fast(p)
        except Exception:
            continue

        label = classify(df.columns)

        org_col   = col(df,"organization","org","company","employer","sponsor","host","agency")
        name_col  = col(df,"name","full name","attendee name","first name")
        last_col  = col(df,"last name","surname","lname")
        email_col = col(df,"email","e-mail","email address").astype(str).str.lower().str.strip()
        event_col = col(df,"event","course","training","session","class","title")
        code_col  = col(df,"code","course code","training code")
        start_col = col(df,"start","start date","date start","begin","start time")
        end_col   = col(df,"end","end date","date end","finish","end time")
        loc_col   = col(df,"location","city","venue","address")
        mode_col  = col(df,"delivery mode","mode","format")
        status    = col(df,"status","registration status","attendance status","state")
        hours     = col(df,"ce hours","hours","ceu")
        desc      = col(df,"description","details","about","notes")
        complete  = col(df,"complete","completed","graduat","certificate","certified").astype(str)

        # name/alias mapping
        base_keys = org_col.apply(norm_key)
        org_ids = base_keys.map(lambda k: org_lookup.get(k,""))

        # domain fallback
        email_domains = email_col.map(pick_domain)
        domain_hits = email_domains.map(lambda d: domain_map.get(d,""))
        org_ids = org_ids.mask(org_ids.eq(""), domain_hits)

        # track unmatched org names (after both passes)
        for k, oid in zip(base_keys, org_ids):
            if k and not oid: unmatched_orgs.add(k)

        # track unmatched contacts (not in contacts master)
        for ek in email_col:
            if ek and ek not in contact_emails: unmatched_contacts.add(ek)

        if label=="catalog":
            for nm, cc, de, hr in zip(event_col, code_col, desc, hours):
                catalog_rows.append({"external_id":hash_id(nm,cc),"Training Name":norm_text(nm),"Code":norm_text(cc),"Description":norm_text(de),"CE Hours Default":norm_text(hr)})
        elif label=="events":
            for nm, st, en, loc, md in zip(event_col, start_col, end_col, loc_col, mode_col):
                events_rows.append({"external_id":hash_id(nm,st,en,loc),"Event Name":norm_text(nm),"Event Type":"training","Start Date":norm_text(st),"End Date":norm_text(en),"Location":norm_text(loc),"Delivery Mode":norm_text(md)})
        elif label=="attendance":
            for fn, ln, em, oid, ev, st in zip(name_col, last_col, email_col, org_ids, event_col, status):
                full = norm_text(fn) or (f"{norm_text(fn)} {norm_text(ln)}").strip()
                parent = parent_map.get(oid,"") if oid else ""
                attendance_rows.append({
                    "external_id":hash_id(em,oid,ev,st),
                    "Contact Email":norm_text(em),
                    "Contact Name":full,
                    "Organization (id)":oid,
                    "Parent Organization (id)":parent,
                    "Event Name":norm_text(ev),
                    "Registration Status":norm_text(st)
                })
        elif label=="certificates":
            for em, tr, dt, comp in zip(email_col, event_col, start_col, complete):
                val = str(comp).lower()
                status_val = "issued" if any(t in val for t in ["yes","true","complete","graduat","1"]) else ""
                cert_rows.append({"external_id":hash_id(em,tr,dt),"Contact Email":norm_text(em),"Training/Event":norm_text(tr),"Issued On":norm_text(dt),"Status":status_val})
        elif label=="payments":
            for oid, amt, payer, tr in zip(col(df,"order id","order","invoice","id"), col(df,"amount","total","payment amount"), email_col, event_col):
                payer_domain = pick_domain(payer)
                payer_org = domain_map.get(payer_domain,"")
                parent = parent_map.get(payer_org,"") if payer_org else ""
                pay_rows.append({
                    "external_id":hash_id(oid,payer,tr,amt),
                    "Order/Invoice":norm_text(oid),
                    "Amount":norm_text(amt),
                    "Payer Email":norm_text(payer),
                    "Payer Org (id)":payer_org,
                    "Parent Organization (id)":parent,
                    "Event/Training":norm_text(tr)
                })

    frames = {
        "training_catalog_preview.csv": pd.DataFrame(catalog_rows).drop_duplicates(subset=["external_id"]),
        "events_preview.csv":           pd.DataFrame(events_rows).drop_duplicates(subset=["external_id"]),
        "attendance_preview.csv":       pd.DataFrame(attendance_rows).drop_duplicates(subset=["external_id"]),
        "certificates_preview.csv":     pd.DataFrame(cert_rows).drop_duplicates(subset=["external_id"]),
        "payments_preview.csv":         pd.DataFrame(pay_rows).drop_duplicates(subset=["external_id"]),
        "unmatched_orgs.csv":           pd.DataFrame(sorted(unmatched_orgs), columns=["org_name_normalized"]),
        "unmatched_contacts.csv":       pd.DataFrame(sorted(unmatched_contacts), columns=["email"]),
    }

    for name, df in frames.items(): df.to_csv(out_dir / name, index=False)

    zip_path = out_dir / (f"cham_normalized_batchA_{'csvonly' if args.csv_only else f'excel_{args.start}'}_.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in frames:
            z.write(out_dir / name, arcname=f"batchA/{name}")

    # quick visibility: show sizes of maps used
    print("MAPS:", {"aliases+canon_keys": len(org_lookup), "domain_map": len(domain_map), "parent_map": len(parent_map)})
    counts = {k: int(v.shape[0]) for k,v in frames.items()}
    print("ZIP:", zip_path)
    print("COUNTS:", counts)
    if not args.csv_only:
        print(f"Processed Excel files {args.start}..{args.start + args.limit - 1}")

if __name__ == "__main__":
    main()
