from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_date(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(text, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def build_dataset(chars_path: Path, returns_path: Path, out_path: Path) -> pd.DataFrame:
    chars = pd.read_csv(chars_path)
    rets = pd.read_csv(returns_path)

    chars.columns = [c.strip() for c in chars.columns]
    rets.columns = [c.strip() for c in rets.columns]

    ret_col = next((c for c in rets.columns if c.upper() == "RET"), None)
    permno_col = next((c for c in rets.columns if c.lower() == "permno"), None)
    date_col = next((c for c in rets.columns if c.upper() == "DATE" or c.lower() == "date"), None)

    if not ret_col or not permno_col or not date_col:
        raise ValueError("Returns file must include permno, DATE, and RET columns.")

    chars["permno"] = pd.to_numeric(chars["permno"], errors="coerce").astype("Int64")
    rets["permno"] = pd.to_numeric(rets[permno_col], errors="coerce").astype("Int64")

    chars["DATE"] = _normalize_date(chars["DATE"])
    rets["DATE"] = _normalize_date(rets[date_col])
    rets["RET"] = pd.to_numeric(rets[ret_col], errors="coerce")

    merged = chars.merge(rets[["permno", "DATE", "RET"]], on=["permno", "DATE"], how="inner")
    merged = merged.dropna(subset=["RET"]).sort_values(["permno", "DATE"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Gu-Kelly-Xiu characteristics with CRSP monthly returns.")
    parser.add_argument("--chars", required=True, help="Path to datashare.csv")
    parser.add_argument("--returns", required=True, help="Path to WRDS/CRSP monthly returns export")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = build_dataset(Path(args.chars), Path(args.returns), Path(args.out))
    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
