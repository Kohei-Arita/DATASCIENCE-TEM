#!/usr/bin/env python3
"""
データ品質レポートを生成

Usage:
  python -m scripts.data_audit --train data/processed/train_processed.csv --cv-splits data/processed/cv_splits.csv --output data/processed/data_audit_report.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def compute_basic_stats(df: pd.DataFrame) -> Dict:
    stats = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "null_counts": df.isnull().sum().to_dict(),
    }
    return stats


def compute_target_stats(df: pd.DataFrame, target_col: str = "aneurysm") -> Dict:
    if target_col not in df.columns:
        return {"available": False}
    y = df[target_col].dropna().values
    if len(y) == 0:
        return {"available": True, "count": 0}
    return {
        "available": True,
        "count": int(len(y)),
        "positive": float(np.mean(y)),
    }


def compute_cv_quality(df: pd.DataFrame, fold_df: pd.DataFrame, target_col: str = "aneurysm") -> Dict:
    if "fold" not in fold_df.columns:
        return {}
    merged = df.join(fold_df["fold"]) if len(fold_df) == len(df) else df.merge(fold_df, left_index=True, right_index=True, how="left")
    fold_counts = merged[merged["fold"] >= 0]["fold"].value_counts().to_dict()
    rate_by_fold = merged[merged["fold"] >= 0].groupby("fold")[target_col].mean().to_dict() if target_col in merged.columns else {}
    return {
        "fold_counts": fold_counts,
        "target_rate_by_fold": {int(k): float(v) for k, v in rate_by_fold.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Create data quality report")
    parser.add_argument("--train", required=True, help="Processed training CSV")
    parser.add_argument("--cv-splits", required=True, help="CV folds CSV")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    logger = setup_logging()

    train_df = pd.read_csv(args.train)
    fold_df = pd.read_csv(args.cv_splits)

    report = {
        "basic": compute_basic_stats(train_df),
        "target": compute_target_stats(train_df, target_col="aneurysm"),
        "cv": compute_cv_quality(train_df, fold_df, target_col="aneurysm"),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved data audit report to {out}")


if __name__ == "__main__":
    main()


