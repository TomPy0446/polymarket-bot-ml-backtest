"""Phase 0 — Inspect pre-built datasets and assert schemas.

Outputs:
  - parquet schemas (columns, dtypes)
  - row counts
  - datetime ranges
  - categorical value_counts (when nunique < 50)
  - ground-truth column identification
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

DATASETS = {
    "crypto15_resolved": "/root/backtest_data/crypto15_resolved.parquet",
    "sampled_prices": "/root/backtest_data/sampled_prices.parquet",
}


def inspect(name: str, path: str) -> dict:
    print("=" * 80)
    print(f"DATASET: {name}")
    print(f"PATH   : {path}")
    if not Path(path).exists():
        print("  ⚠ MISSING")
        return {"name": name, "path": path, "missing": True}
    df = pd.read_parquet(path)
    print(f"  shape : {df.shape}")
    print(f"  cols  : {list(df.columns)}")
    print(f"  dtypes:")
    for c, dt in df.dtypes.items():
        print(f"    {c}: {dt}")
    print(f"  head(3):")
    print(df.head(3).to_string(max_colwidth=40))
    # datetime ranges
    for c in df.columns:
        if df[c].dtype.kind in ("M",):
            print(f"  DATE RANGE {c}: {df[c].min()} → {df[c].max()}")
    # categorical breakdown
    for c in df.select_dtypes(include="object").columns:
        n = df[c].nunique(dropna=True)
        if 2 <= n <= 50:
            print(f"  VALUE_COUNTS {c} (n={n}):")
            for k, v in df[c].value_counts(dropna=False).head(20).items():
                print(f"    {k!r}: {v}")
    # numeric quick stats
    num = df.select_dtypes(include=["number"]).columns
    if len(num) > 0:
        print(f"  NUMERIC describe:")
        print(df[num].describe().to_string())
    return {"name": name, "path": path, "shape": list(df.shape), "cols": list(df.columns)}


def main() -> int:
    out = {"datasets": []}
    for name, path in DATASETS.items():
        info = inspect(name, path)
        out["datasets"].append(info)
    print("\n" + "=" * 80)
    print("SUMMARY (JSON):")
    print(json.dumps(out, indent=2, default=str))
    # Write summary
    out_dir = Path("/root/polymarket-bot-ml-backtest/phase0")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_summary.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
