"""Prepare le CSV Adult Income pour le pipeline mlproject."""
from __future__ import annotations

from mlproject.config import CATEGORICAL_FEATURES, DATA_PATH, RAW_DATA_PATH
from mlproject.data import prepare_raw


def prepare(raw_path=RAW_DATA_PATH, out_path=DATA_PATH):
    import pandas as pd

    df = prepare_raw(pd.read_csv(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] {len(df)} lignes ecrites -> {out_path}")
    return df


if __name__ == "__main__":
    prepare()
