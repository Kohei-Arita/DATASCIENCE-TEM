#!/usr/bin/env python3
"""
KaggleからRSNAコンペデータをダウンロード

Usage:
    python -m scripts.download_data --competition rsna-intracranial-aneurysm-detection --output data/raw --unzip
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def ensure_kaggle_credentials(logger: logging.Logger) -> None:
    """Kaggle APIの資格情報が設定されているか簡易チェック"""
    import os

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if username and key:
        logger.info("Kaggle credentials found in environment variables")
        return

    if kaggle_json.exists():
        logger.info(f"Kaggle credentials found at {kaggle_json}")
        return

    logger.warning("Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY or place kaggle.json under ~/.kaggle/")


def download_kaggle_data(competition: str, output_dir: str, unzip: bool = True) -> bool:
    """Kaggle APIでデータをダウンロード"""
    logger = setup_logging()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ensure_kaggle_credentials(logger)

    logger.info(f"Downloading competition='{competition}' to '{output_dir}'")

    try:
        cmd = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition,
            "-p",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            return False

        logger.info("Download completed successfully")

        if unzip:
            import zipfile

            zip_files = list(output_path.glob("*.zip"))
            if not zip_files:
                logger.warning("No zip files found to extract. Perhaps files are already extracted.")
            for zip_file in zip_files:
                logger.info(f"Extracting {zip_file}")
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(output_path)
                zip_file.unlink()

        files = [f.name for f in output_path.iterdir()]
        logger.info(f"Files in output dir: {files}")

        return True

    except FileNotFoundError:
        logger.error("'kaggle' CLI not found. Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle competition data")
    parser.add_argument("--competition", required=True, help="Competition name (e.g., rsna-intracranial-aneurysm-detection)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-unzip", action="store_true", help="Do not unzip downloaded files")

    args = parser.parse_args()
    success = download_kaggle_data(args.competition, args.output, unzip=not args.no_unzip)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


