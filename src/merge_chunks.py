#!/usr/bin/env python3
import zipfile, io, pandas as pd
from pathlib import Path
from pandas.errors import EmptyDataError, ParserError

OUT = Path("out")
zips = sorted(OUT.glob("cham_normalized_batchA_*.zip"))

# target file names and the keys we dedupe on
TARGETS = [
    ("training_catalog_preview.csv", ["external_id"]),
    ("events_preview.csv",           ["external_id"]),
    ("attendance_preview.csv",       ["external_id"]),
    ("certificates_preview.csv",     ["external_id"]),
    ("payments_preview.csv",         ["external_id"]),
    ("unmatched_orgs.csv",           None),
    ("unmatched_contacts.csv",       None),
]

frames = {name: [] for name, _ in TARGETS}
cols_seen: dict[str, list[str]] = {}   # remember columns we saw per file name
skipped = []                            # [(zipname, arcname, reason), ...]

def read_csv_lenient(fobj, name: str) -> pd.DataFrame:
    """Read CSV from file-like, tolerate empty/bad files. Returns empty df if needed."""
    # First attempt: straightforward
    try:
        df = pd.read_csv(fobj, dtype=str)
        return df.fillna("")
    except EmptyDataError:
        # truly empty — return empty df with any known columns for this file type
        if name in cols_seen:
            return pd.DataFrame(columns=cols_seen[name])
        return pd.DataFrame()
    except ParserError:
        # try latin1 as a fallback
        try:
            fobj.seek(0)
        except Exception:
            pass
        try:
            buf = io.BytesIO(fobj.read())
            df = pd.read_csv(buf, dtype=str, encoding="latin1")
            return df.fillna("")
        except Exception:
            # give up; caller will decide to skip
            raise

for zpath in zips:
    with zipfile.ZipFile(zpath, "r") as z:
        names = set(z.namelist())
        for fname, _keys in TARGETS:
            arc = f"batchA/{fname}"
            if arc not in names:
                skipped.append((zpath.name, arc, "missing"))
                continue
            # If the entry is zero length, skip quickly
            info = z.getinfo(arc)
            if info.file_size == 0:
                skipped.append((zpath.name, arc, "empty (0 bytes)"))
                continue
            with z.open(arc) as f:
                try:
                    df = read_csv_lenient(f, fname)
                except Exception as e:
                    skipped.append((zpath.name, arc, f"parse error: {type(e).__name__}"))
                    continue
                if df is None or df.shape[0] == 0 and df.shape[1] == 0:
                    skipped.append((zpath.name, arc, "empty dataframe"))
                    continue
                # remember columns for this file type for future empty files
                cols_seen.setdefault(fname, list(df.columns))
                # normalize to strings and collect
                frames[fname].append(df.astype(str).fillna(""))

# Merge & dedupe
merged = {}
for fname, keys in TARGETS:
    if not frames[fname]:
        merged[fname] = pd.DataFrame(columns=cols_seen.get(fname, []))
        continue
    df = pd.concat(frames[fname], ignore_index=True)
    if keys:
        df = df.drop_duplicates(subset=keys)
    else:
        df = df.drop_duplicates()
    merged[fname] = df

# Write outputs
final_dir = OUT / "merged"; final_dir.mkdir(exist_ok=True)
for fname, df in merged.items():
    (final_dir / fname).write_text(df.to_csv(index=False))

final_zip = OUT / "cham_normalized_batchA_merged.zip"
with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for fname in merged:
        z.write(final_dir / fname, arcname=f"batchA/{fname}")

print("Merged ZIP:", final_zip)
for fname, df in merged.items():
    print(f"{fname}: {len(df)} rows")

# Helpful note about skips
if skipped:
    print("\nSkipped entries:")
    for zname, arc, reason in skipped[:20]:
        print(f" - {zname} :: {arc} -> {reason}")
    if len(skipped) > 20:
        print(f" ... and {len(skipped)-20} more")
