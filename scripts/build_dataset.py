# scripts/build_dataset.py

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Make scripts/ importable when running this file directly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

import clean  # type: ignore
import feature_extract  # type: ignore
import normalize  # type: ignore


RAW_PATH = PROJECT_ROOT / "data" / "synthetic" / "sessions.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")

    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def main():
    print(f"[*] Loading raw dataset from: {RAW_PATH}")
    df_raw = load_raw(RAW_PATH)
    print(f"[*] Raw shape: {df_raw.shape}")

    # ---------- CLEAN ----------
    print("[*] Cleaning dataset (nulls + outliers)...")
    df_clean = clean.clean_dataframe(
        df_raw,
        exclude_numeric=["label", "session_id", "user_id"],
        exclude_categorical=[],
        iqr_multiplier=1.5,
    )
    print(f"[*] After clean: {df_clean.shape}")

    # ---------- FEATURE ENGINEERING ----------
    print("[*] Applying feature engineering (behavior + device + network)...")
    df_feat = feature_extract.build_features(df_clean)
    print(f"[*] After feature engineering: {df_feat.shape}")

    # ---------- TRAIN/TEST SPLIT ----------
    print("[*] Splitting into train/test...")
    if "label" in df_feat.columns:
        y = df_feat["label"]
        stratify = y
    else:
        y = None
        stratify = None

    train_df, test_df = train_test_split(
        df_feat,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    train_path_raw = PROCESSED_DIR / "train_raw.parquet"
    test_path_raw = PROCESSED_DIR / "test_raw.parquet"

    train_df.to_parquet(train_path_raw, index=False)
    test_df.to_parquet(test_path_raw, index=False)

    print(f"[*] Saved intermediate train_raw → {train_path_raw}")
    print(f"[*] Saved intermediate test_raw → {test_path_raw}")

    # ---------- SCALING ----------
    print("[*] Scaling numeric features with shared scaler (train/test)...")

    # We will not scale IDs / label
    exclude_numeric = ["label", "session_id", "user_id"]

    # Fit scaler on train
    train_scaled, scaler, numeric_cols = normalize.fit_and_transform(
        train_df,
        method="standard",
        exclude_numeric=exclude_numeric,
    )

    # Transform test with same scaler (no leakage)
    test_scaled = normalize.transform_with_existing(
        test_df,
        scaler,
        numeric_cols,
    )

    train_out = PROCESSED_DIR / "train.parquet"
    test_out = PROCESSED_DIR / "test.parquet"

    train_scaled.to_parquet(train_out, index=False)
    test_scaled.to_parquet(test_out, index=False)

    # Save scaler object for API / later models
    scaler_obj = {
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "method": "standard",
    }
    scaler_path = ARTIFACTS_DIR / "tabular_scaler.pkl"
    joblib.dump(scaler_obj, scaler_path)

    print(f"✅ Final processed train → {train_out}")
    print(f"✅ Final processed test  → {test_out}")
    print(f"✅ Saved scaler → {scaler_path}")
    print("[✓] Day 7 pipeline complete.")


if __name__ == "__main__":
    main()
