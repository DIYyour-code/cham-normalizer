#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

DATA = Path("data")
infile = DATA/"contact_domain_to_winner.xlsx"  # rename if yours differs
outfile = DATA/"domain_to_org.csv"

df = pd.read_excel(infile, dtype=str).fillna("")

# choose domain: prefer DomainPrimaryNorm, else EmailDomain
dom = df.get("DomainPrimaryNorm", pd.Series([""]*len(df)))
dom = dom.where(dom.str.strip().ne(""), df.get("EmailDomain","").astype(str))
dom = dom.str.strip().str.lower().str.replace(r"^www\.", "", regex=True)

# target org id
target = df.get("WinnerOrgId","").astype(str).str.strip()

map_df = pd.DataFrame({"Domain": dom, "To Org (id)": target})

# drop free-mail / ISP domains
free = {
    "gmail.com","yahoo.com","outlook.com","hotmail.com","aol.com","icloud.com",
    "comcast.net","att.net","msn.com","live.com","proton.me","protonmail.com",
    "ymail.com","me.com","mail.com","gmx.com","cox.net","sbcglobal.net"
}
map_df = map_df[map_df["Domain"].ne("") & map_df["To Org (id)"].ne("") & ~map_df["Domain"].isin(free)]

# dedupe: if a domain appears multiple times, keep the first non-empty WinnerOrgId
map_df = map_df.drop_duplicates(subset=["Domain"], keep="first").sort_values("Domain")

outfile.parent.mkdir(exist_ok=True, parents=True)
map_df.to_csv(outfile, index=False)
print(f"Wrote {outfile} with {len(map_df)} rows.")
