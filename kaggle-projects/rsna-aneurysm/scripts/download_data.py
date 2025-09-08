#!/usr/bin/env python3
"""
KaggleからRSNAコンペデータをダウンロード

Usage:
    python -m scripts.download_data --competition rsna-intracranial-aneurysm-detection --output data/raw --unzip
"""

import argparse
import subprocess
import sys
import shlex
from pathlib import Path
import logging
import os
from .security_utils import run_kaggle_command, validate_competition_name, validate_file_path


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
    """
    Kaggle APIでデータをダウンロード (secure version)
    
    Args:
        competition: Kaggle competition name (must be alphanumeric with dashes)
        output_dir: Output directory path
        unzip: Whether to unzip downloaded files
        
    Returns:
        bool: Success status
        
    Raises:
        SecurityError: If inputs fail validation
        FileNotFoundError: If kaggle CLI is not installed
    """
    logger = setup_logging()
    
    # Secure input validation
    try:
        competition = validate_competition_name(competition)
        output_path = validate_file_path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        return False

    ensure_kaggle_credentials(logger)

    logger.info(f"Downloading competition='{competition}' to '{output_dir}'")

    try:
        # Use secure kaggle command execution
        args = [
            "competitions",
            "download",
            "-c",
            competition,
            "-p",
            str(output_path),
        ]

        result = run_kaggle_command(args, cwd=output_path)

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

    except subprocess.TimeoutExpired:
        logger.error("Download timed out")
        return False
    except FileNotFoundError:
        logger.error("'kaggle' CLI not found. Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
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


