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


def download_essential_files(competition: str, output_path: Path, logger: logging.Logger) -> bool:
    """Download essential CSV files first (quick download)"""
    essential_files = ["train.csv", "train_localizers.csv"]
    
    for file_name in essential_files:
        logger.info(f"Downloading essential file: {file_name}")
        try:
            args = [
                "competitions",
                "download", 
                "-c", 
                competition,
                "-f",
                file_name,
                "-p", 
                str(output_path),
                "--force"
            ]
            result = run_kaggle_command(args, cwd=output_path, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded {file_name}")
            else:
                logger.warning(f"Could not download {file_name}: {result.stderr}")
        except Exception as e:
            logger.warning(f"Failed to download {file_name}: {e}")
    
    # Try non-essential files best-effort (no failure)
    for optional_file in ["test.csv", "sample_submission.csv"]:
        logger.info(f"(optional) Trying to download: {optional_file}")
        try:
            args = [
                "competitions",
                "download",
                "-c",
                competition,
                "-f",
                optional_file,
                "-p",
                str(output_path),
                "--force"
            ]
            result = run_kaggle_command(args, cwd=output_path, timeout=60)
            if result.returncode == 0:
                logger.info(f"Successfully downloaded optional file: {optional_file}")
            else:
                logger.warning(f"Optional file not available: {optional_file}")
        except Exception as e:
            logger.warning(f"Optional download failed for {optional_file}: {e}")

    return True


def download_kaggle_data(competition: str, output_dir: str, unzip: bool = True, files_only: bool = False) -> bool:
    """
    Kaggle APIでデータをダウンロード (secure version with selective download)
    
    Args:
        competition: Kaggle competition name (must be alphanumeric with dashes)
        output_dir: Output directory path
        unzip: Whether to unzip downloaded files
        files_only: If True, download only essential CSV files (fast)
        
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
        # First, download essential CSV files quickly
        download_essential_files(competition, output_path, logger)
        
        # If only essential files requested, return early
        if files_only:
            files = [f.name for f in output_path.iterdir()]
            logger.info(f"Essential files downloaded: {files}")
            return True
        
        # Download large image files with extended timeout
        logger.info("Downloading large image datasets (this may take 10+ minutes)...")
        args = [
            "competitions",
            "download",
            "-c",
            competition,
            "-f",
            "train_images.zip",
            "-p",
            str(output_path),
            "--force"
        ]

        # Use extended timeout for large files (30 minutes)
        result = run_kaggle_command(args, cwd=output_path, timeout=1800)

        if result.returncode != 0:
            logger.error(f"Large file download failed: {result.stderr}")
            logger.info("Essential files are still available for initial development")
            return True  # Still return True since we have essential files

        logger.info("Large file download completed successfully")

        if unzip:
            import zipfile

            zip_files = list(output_path.glob("*.zip"))
            if not zip_files:
                logger.warning("No zip files found to extract.")
            else:
                for zip_file in zip_files:
                    logger.info(f"Extracting {zip_file} (this may take several minutes)...")
                    with zipfile.ZipFile(zip_file, "r") as zip_ref:
                        zip_ref.extractall(output_path)
                    logger.info(f"Extraction completed for {zip_file}")
                    zip_file.unlink()

        files = [f.name for f in output_path.iterdir()]
        logger.info(f"Files in output dir: {files}")

        return True

    except subprocess.TimeoutExpired:
        logger.error("Download timed out - but essential CSV files may be available")
        files = [f.name for f in output_path.iterdir()]
        if files:
            logger.info(f"Available files: {files}")
            return True
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
    parser.add_argument("--files-only", action="store_true", help="Download only essential CSV files (fast, no images)")

    args = parser.parse_args()
    success = download_kaggle_data(
        args.competition, 
        args.output, 
        unzip=not args.no_unzip,
        files_only=args.files_only
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


