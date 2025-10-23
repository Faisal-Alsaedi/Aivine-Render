"""
Download and extract the Kaggle dataset into:
- data/raw/                       (downloaded zips)
- data/dataset_flat_structure/    (train/validation/test)

Uses Kaggle Python API (no subprocess). Requires a valid kaggle.json.
Windows:
  - Default: %USERPROFILE%\\.kaggle\\kaggle.json
  - Or set env KAGGLE_CONFIG_DIR to a folder that contains kaggle.json
"""

from pathlib import Path
import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# local config paths
from config import RAW_DIR, SPLIT_DIR

KAGGLE_DATASET = "pappu54/indoor-plant-disease-dataset-23-classes"

def _assert_creds():
    # Kaggle API checks for kaggle.json internally, but we fail fast with a clear error.
    cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    default_path = Path.home() / ".kaggle" / "kaggle.json"
    if cfg_dir:
        cfg_path = Path(cfg_dir) / "kaggle.json"
    else:
        cfg_path = default_path
    if not cfg_path.exists():
        raise RuntimeError(
            f"Missing kaggle.json. Put it at {cfg_path} or set KAGGLE_CONFIG_DIR to the folder containing it."
        )

def _ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def _extract_first_zip(zip_path: Path, dest: Path):
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

def _locate_split_root(search_root: Path) -> Path:
    """
    Try to find 'dataset_flat_structure' containing train/validation/test.
    """
    candidates = list(search_root.rglob("dataset_flat_structure"))
    for c in candidates:
        if (c / "train").exists() and (c / "validation").exists() and (c / "test").exists():
            return c
    raise RuntimeError("dataset_flat_structure with train/validation/test not found after extraction.")

def main():
    _assert_creds()
    _ensure_dirs()

    print("Authenticating with Kaggle API ...")
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    # download as zipped files into RAW_DIR
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(RAW_DIR),
        force=True,    # overwrite if exists
        quiet=False
    )

    # Kaggle API bundles everything into a single archive named after the dataset owner/title
    # Find the newest .zip in RAW_DIR
    zips = sorted(RAW_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise RuntimeError("Download reported success, but no .zip found in data/raw.")
    main_zip = zips[0]

    # Extract top-level archive into RAW_DIR/extracted
    extracted_root = RAW_DIR / "extracted"
    if extracted_root.exists():
        shutil.rmtree(extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)

    _extract_first_zip(main_zip, extracted_root)

    # Locate dataset_flat_structure and copy it to SPLIT_DIR
    src_split = _locate_split_root(extracted_root)
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    shutil.copytree(src_split, SPLIT_DIR)

    print(f"Dataset ready at: {SPLIT_DIR}")
    print("Done.")

if __name__ == "__main__":
    main()