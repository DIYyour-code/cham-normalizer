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

def pick_org_col(df):
    # Score columns: prefer exact org/organization, penalize address-like
    addr_bad = {"address","street","city","state","zip","postal","phone"}
    scores = []
    for c in df.columns:
        cl = str(c).lower().strip()
        score = 0
        if cl in {"organization","org"}: score += 100
        if "organization" in cl or cl == "org": score += 10
        if any(b in cl for b in addr_bad): score -= 50
        scores.append((score, c))
    best = max(scores)[1]
    return df[best]

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
    lookup = {norm_key(r["Name (canonical)"]): r["org_id"] for _, r in canon.iterrows()}
    if alias_path and alias_path.exists():
        alias_df = pd.read_csv(alias_path, dtype=str).fillna("")
        for _, r in alias_df.iterrows():
            if r.get("Alias") and r.get("To Org (id)"):
                lookup[norm_key(r["Alias"])] = r["To Org (id)"]
    return lookup

def build_contact_email_set(contacts_xlsx: Path):
    xl = pd.ExcelFile(contacts_xlsx)
    pick = next((s for s in xl.sheet_names if "email" in " ".join(map(str, xl.parse(s, nrows=1).columns)).lower()), xl.sheet_names[0])
    df = xl.parse(pick, dtype=str).fillna("")
    email_col = next((c for c in df.columns if "email" in c.lower()), df.columns[0])
    return {e for e in df[email_col].astype(str).str.lower().str.strip() if e}

def pick_org_col(df):
    # Prefer exact org columns; penalize address-like columns
    addr_bad = {"address","street","line1","line2","city","state","zip","postal","phone"}
    best = None; best_score = -10**9
    for c in df.columns:
        cl = str(c).lower().strip()
        score = 0
        if cl in {"organization","org","employer","sponsor","host","agency","company"}: score += 100
        if "organization" in cl or cl == "org": score += 20
        if any(b in cl for b in addr_bad): score -= 200
        if "contact" in cl or "email" in cl: score -= 50
        if score > best_score: best_score, best = score, c
    return df[best] if best is not None else pd.Series([""]*len(df))

def process_files(files, org_lookup, contact_emails, csv_mode=False):
    catalog_rows, events_rows, attendance_rows, cert_rows, pay_rows = [], [], [], [], []
    unmatched_orgs, unmatched_contacts = set(), set()

    for p in files:
        try:
            if csv_mode:
                df = read_csv_fast(p)
            else:
                df = read_excel_best_sheet(p) if p.suffix.lower() in (".xlsx",".xls") else read_csv_fast(p)
        except Exception:
            continue

        label = classify(df.columns)
        org_col   = pick_org_col(df)
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

        org_ids = org_col.apply(norm_key).map(lambda k: org_lookup.get(k,""))
        for k, oid in zip(org_col.apply(norm_key), org_ids):
            if k and not oid: unmatched_orgs.add(k)
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
                attendance_rows.append({"external_id":hash_id(em,oid,ev,st),"Contact Email":norm_text(em),"Contact Name":full,"Organization (id)":oid,"Event Name":norm_text(ev),"Registration Status":norm_text(st)})
        elif label=="certificates":
            for em, tr, dt, comp in zip(email_col, event_col, start_col, complete):
                val = str(comp).lower()
                status_val = "issued" if any(t in val for t in ["yes","true","complete","graduat","1"]) else ""
                cert_rows.append({"external_id":hash_id(em,tr,dt),"Contact Email":norm_text(em),"Training/Event":norm_text(tr),"Issued On":norm_text(dt),"Status":status_val})
        elif label=="payments":
            for oid, amt, payer, tr in zip(col(df,"order id","order","invoice","id"), col(df,"amount","total","payment amount"), email_col, event_col):
                pay_rows.append({"external_id":hash_id(oid,payer,tr,amt),"Order/Invoice":norm_text(oid),"Amount":norm_text(amt),"Payer Email":norm_text(payer),"Event/Training":norm_text(tr)})

    frames = {
        "training_catalog_preview.csv": pd.DataFrame(catalog_rows).drop_duplicates(subset=["external_id"]),
        "events_preview.csv":           pd.DataFrame(events_rows).drop_duplicates(subset=["external_id"]),
        "attendance_preview.csv":       pd.DataFrame(attendance_rows).drop_duplicates(subset=["external_id"]),
        "certificates_preview.csv":     pd.DataFrame(cert_rows).drop_duplicates(subset=["external_id"]),
        "payments_preview.csv":         pd.DataFrame(pay_rows).drop_duplicates(subset=["external_id"]),
        "unmatched_orgs.csv":           pd.DataFrame(sorted(unmatched_orgs), columns=["org_name_normalized"]),
        "unmatched_contacts.csv":       pd.DataFrame(sorted(unmatched_contacts), columns=["email"]),
    }
    return frames

def main():
    ap = argparse.ArgumentParser(description="CHAM Batch A normalizer (local, chunked).")
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

    org_lookup     = build_org_lookup(canon_path, alias_path if alias_path.exists() else None)
    contact_emails = build_contact_email_set(contacts_xlsx)

    zip1 = data_dir / "events data exports.zip"
    zip2 = data_dir / "New Training Data 2.zip"

    csv_files, xls_files = [], []
    for z in [zip1, zip2]:
        if z.exists():
            csv_files += unzip_select(z, [".csv"], work_dir)
            xls_files += unzip_select(z, [".xlsx",".xls"], work_dir)

    files = csv_files if args.csv_only else (csv_files + xls_files[args.start: args.start + args.limit])

    frames = process_files(files, org_lookup, contact_emails, csv_mode=args.csv_only)

    for name, df in frames.items():
        df.to_csv(out_dir / name, index=False)

    zip_path = out_dir / (f"cham_normalized_batchA_{'csvonly' if args.csv_only else f'excel_{args.start}'}_.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in frames:
            z.write(out_dir / name, arcname=f"batchA/{name}")

    counts = {k: int(v.shape[0]) for k,v in frames.items()}
    print("ZIP:", zip_path)
    print("COUNTS:", counts)
    if not args.csv_only:
        print(f"Processed Excel files {args.start}..{args.start + args.limit - 1}")

if __name__ == "__main__":
    main()
