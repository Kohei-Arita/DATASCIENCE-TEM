#!/usr/bin/env python3
"""
患者層化Group KFoldなど、RSNA向けのCV分割を作成

Usage:
  python -m scripts.make_folds --config configs/cv.yaml --data data/processed/train_processed.csv --output data/processed/cv_splits.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, GroupKFold


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def read_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_stratified_group_folds(df: pd.DataFrame, target_col: str, group_col: str, n_splits: int, seed: int) -> pd.Series:
    """層化かつグループ制約を同時に満たす近似法（反復割当）"""
    rng = np.random.default_rng(seed)
    fold = np.full(len(df), -1, dtype=int)

    # グループ単位に集約し、グループのターゲット比率で層化
    grp = df.groupby(group_col)[target_col].agg(["count", "sum"]).reset_index()
    grp["pos_ratio"] = (grp["sum"] / grp["count"]).fillna(0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # 比率をビン化して層化の安定性を向上
    bins = np.minimum((grp["pos_ratio"] * 10).round(0), 9)

    for f, (_, valid_idx) in enumerate(skf.split(grp, bins)):
        valid_groups = set(grp.iloc[valid_idx][group_col].values)
        is_valid = df[group_col].isin(valid_groups).values
        # 既に割り当て済みの行は上書きしない
        fold = np.where((fold == -1) & is_valid, f, fold)

    # 割当漏れがあればランダムに補完
    unassigned = np.where(fold == -1)[0]
    if len(unassigned) > 0:
        logging.warning(f"{len(unassigned)} samples were unassigned; randomly assigning to folds")
        fold[unassigned] = rng.integers(0, n_splits, size=len(unassigned))

    return pd.Series(fold, name="fold")


def create_cv(df: pd.DataFrame, cfg: dict) -> pd.Series:
    n_folds = cfg.get("n_folds", 5)
    seed = cfg.get("seed", 42)
    method = cfg.get("cv_method", "stratified_group_kfold")

    strat_target = cfg.get("stratification", {}).get("target_column", "aneurysm")
    group_col = cfg.get("grouping", {}).get("group_column", "PatientID")

    if method == "stratified_group_kfold":
        return get_stratified_group_folds(df, strat_target, group_col, n_folds, seed)
    elif method == "group_kfold":
        gkf = GroupKFold(n_splits=n_folds)
        fold = np.full(len(df), -1, dtype=int)
        for f, (_, v_idx) in enumerate(gkf.split(df, groups=df[group_col])):
            fold[v_idx] = f
        return pd.Series(fold, name="fold")
    elif method == "stratified_kfold":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold = np.full(len(df), -1, dtype=int)
        for f, (_, v_idx) in enumerate(skf.split(df, df[strat_target])):
            fold[v_idx] = f
        return pd.Series(fold, name="fold")
    else:
        raise ValueError(f"Unsupported cv_method: {method}")


def quality_check(df: pd.DataFrame, fold_col: str, target_col: str) -> Tuple[float, float]:
    """Foldのサイズバランスとターゲット率の標準偏差を返す"""
    size_std = df[df[fold_col] >= 0][fold_col].value_counts().std()
    rates = df[df[fold_col] >= 0].groupby(fold_col)[target_col].mean().values
    rate_std = float(np.std(rates)) if len(rates) > 0 else 0.0
    return float(size_std), rate_std


def main():
    parser = argparse.ArgumentParser(description="Create CV splits for RSNA")
    parser.add_argument("--config", required=True, help="Path to CV config YAML")
    parser.add_argument("--data", required=True, help="Input processed CSV with targets and groups")
    parser.add_argument("--output", required=True, help="Output CSV path to save folds")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = read_config(Path(args.config))

    df = pd.read_csv(args.data)
    logger.info(f"Loaded data: shape={df.shape}")

    fold_series = create_cv(df, cfg)
    df["fold"] = fold_series

    size_std, rate_std = quality_check(df, "fold", cfg.get("stratification", {}).get("target_column", "aneurysm"))
    logger.info(f"Fold size std: {size_std:.3f}, target rate std: {rate_std:.5f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["fold"]].to_csv(out_path, index=False)
    logger.info(f"Saved folds to {out_path}")


if __name__ == "__main__":
    main()


