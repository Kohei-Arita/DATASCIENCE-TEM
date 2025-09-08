#!/usr/bin/env python3
"""
RSNA: train.csv と DICOMメタデータを結合して学習用メタデータを生成

Usage:
    python -m scripts.create_metadata --train-csv data/raw/train.csv --metadata data/processed/train_metadata.csv --output data/processed/train_processed.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 代表的な列名の標準化（存在すれば）
    col_map = {
        "StudyInstanceUID": "StudyInstanceUID",
        "SeriesInstanceUID": "SeriesInstanceUID",
        "SOPInstanceUID": "SOPInstanceUID",
        "PatientID": "PatientID",
        "aneurysm": "aneurysm",
        "label": "aneurysm",
        "target": "aneurysm",
        "image_id": "image_id",
        "ID": "image_id",
    }
    # 実在する列のみを置換
    rename_map = {c: col_map[c] for c in df.columns if c in col_map}
    return df.rename(columns=rename_map)


def synthesize_image_id(df: pd.DataFrame) -> pd.DataFrame:
    # Kaggle RSNAの典型: 画像IDをSOPInstanceUIDとみなすケース。
    if "image_id" not in df.columns:
        if "SOPInstanceUID" in df.columns:
            df["image_id"] = df["SOPInstanceUID"].astype(str)
        elif "ID" in df.columns:
            df["image_id"] = df["ID"].astype(str)
    return df


def create_training_metadata(train_csv: Path, dicom_meta_csv: Path, output_csv: Path) -> pd.DataFrame:
    logger = setup_logging()
    logger.info(f"Loading train CSV: {train_csv}")
    train_df = load_csv_safe(train_csv)
    train_df = normalize_columns(train_df)

    logger.info(f"Loading DICOM metadata: {dicom_meta_csv}")
    meta_df = load_csv_safe(dicom_meta_csv)
    meta_df = normalize_columns(meta_df)

    # 画像IDの同定
    train_df = synthesize_image_id(train_df)
    meta_df = synthesize_image_id(meta_df)

    # 結合キーの推定: image_id優先、なければSOPInstanceUID
    join_keys = None
    for key in ["image_id", "SOPInstanceUID"]:
        if key in train_df.columns and key in meta_df.columns:
            join_keys = key
            break
    if join_keys is None:
        raise ValueError("No common key to join between train and metadata.")

    logger.info(f"Merging on key: {join_keys}")
    merged = train_df.merge(meta_df, on=join_keys, how="left", suffixes=("", "_meta"))

    # ターゲット列のチェック
    if "aneurysm" not in merged.columns:
        raise ValueError("Target column 'aneurysm' not found after merge.")

    # 保存
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    logger.info(f"Saved processed metadata: {output_csv} shape={merged.shape}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Create merged training metadata")
    parser.add_argument("--train-csv", required=True, help="Path to train.csv")
    parser.add_argument("--metadata", required=True, help="Path to extracted DICOM metadata CSV")
    parser.add_argument("--output", required=True, help="Path to save processed metadata CSV")
    args = parser.parse_args()

    create_training_metadata(Path(args.train_csv), Path(args.metadata), Path(args.output))


if __name__ == "__main__":
    main()


