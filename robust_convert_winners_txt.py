# robust_convert_winners_txt.py
import re, pandas as pd
from datetime import datetime
from pathlib import Path

INPUT_TXT  = "recent_winners.txt"                      # your raw text (reverse-ordered is fine)
OUTPUT_CSV = "DC5_Recent_Winners_full_converted.csv"   # app-ready CSV
KEEP_KEYWORD = None  # e.g. "Midday" to keep only midday lines; or None to keep all

def parse_date_any(s: str):
    s = s.strip().replace("–","-").replace("—","-")
    # ISO yyyy-mm-dd
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except: pass
    # mm/dd/yyyy
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", s)
    if m:
        mo, d, y = map(int, m.groups())
        try:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except: pass
    # textual
    fmts = [
        "%a, %b %d, %Y", "%A, %b %d, %Y", "%a, %B %d, %Y", "%A, %B %d, %Y",
        "%b %d, %Y", "%B %d, %Y"
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except: pass
    return None

src = Path(INPUT_TXT)
rows, skipped = [], []
with src.open("r", encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        if KEEP_KEYWORD and KEEP_KEYWORD.lower() not in line.lower():
            continue
        # grab the LAST 5-digit token in the line
        matches = list(re.finditer(r"(?<!\d)(\d{5})(?!\d)", line))
        if not matches:
            skipped.append(line)
            continue
        result = matches[-1].group(1)
        # look for date anywhere in the line (or before the result)
        date_guess = line[:matches[-1].start()].strip()
        d = parse_date_any(date_guess) or parse_date_any(line) or date_guess
        rows.append({"Date": d, "Result": result})

df = pd.DataFrame(rows)

# Normalize Result, de-dupe, sort oldest->newest
if not df.empty:
    df["Result"] = df["Result"].astype(str).str.replace(r"\D","", regex=True).str.zfill(5)
    df = df.drop_duplicates(subset=["Date", "Result"], keep="last")
    try:
        df["_d"] = pd.to_datetime(df["Date"], errors="raise")
        df = df.sort_values("_d").drop(columns=["_d"]).reset_index(drop=True)
    except:
        # if some dates are unparseable strings, keep input order
        df = df.reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)

print(f"Done. Wrote {len(df)} rows to {OUTPUT_CSV}")
if skipped:
    print(f"Note: {len(skipped)} lines had no 5-digit result and were skipped.")
