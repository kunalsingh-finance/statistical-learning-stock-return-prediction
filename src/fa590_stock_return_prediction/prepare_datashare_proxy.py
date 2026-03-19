from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd


def normalize_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def build_proxy_dataset(zip_path: Path, out_path: Path, max_permnos: int) -> pd.DataFrame:
    selected_permnos: set[int] = set()
    chunks = []

    read_kwargs = {"chunksize": 100_000, "dtype": {"permno": "Int64", "DATE": "string"}, "low_memory": False}

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("datashare.csv") as raw:
            text_stream = io.TextIOWrapper(raw, encoding="utf-8")
            for chunk in pd.read_csv(text_stream, **read_kwargs):
                chunk["DATE_num"] = pd.to_numeric(chunk["DATE"], errors="coerce")
                chunk = chunk[(chunk["DATE_num"] >= 20090101) & (chunk["DATE_num"] <= 20211231)].copy()
                if chunk.empty:
                    continue

                for permno in chunk["permno"].dropna().astype(int).unique():
                    if len(selected_permnos) >= max_permnos and permno not in selected_permnos:
                        continue
                    selected_permnos.add(int(permno))

                chunk = chunk[chunk["permno"].isin(selected_permnos)].copy()
                if chunk.empty:
                    continue

                chunk["RET"] = pd.to_numeric(chunk["mom1m"], errors="coerce")
                chunk = chunk.dropna(subset=["RET"])
                chunk["DATE"] = normalize_date(chunk["DATE_num"])
                chunk = chunk.drop(columns=["mom1m", "DATE_num"])
                chunks.append(chunk)

    if not chunks:
        raise ValueError("No rows were extracted from datashare.zip for the requested period.")

    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=["permno", "DATE", "RET"]).sort_values(["permno", "DATE"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a public proxy return-prediction dataset from datashare.zip.")
    parser.add_argument("--zip-path", required=True, help="Path to datashare.zip")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--max-permnos", type=int, default=600, help="Maximum number of stocks to keep for the proxy dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_proxy_dataset(Path(args.zip_path), Path(args.out), args.max_permnos)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Stocks: {df['permno'].nunique():,}")
    print(f"Dates: {df['DATE'].min()} to {df['DATE'].max()}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
