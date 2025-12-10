import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib


# 👇 using YOUR file here
DATA_PATH = Path("data/training_dataset.csv")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found.\n"
            "Make sure data/training_dataset.csv exists."
        )
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_xy(df: pd.DataFrame):
    # 👇 this must match your target column name
    target_col = "isFraud"

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataset. "
            f"Columns available: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_names = X.columns.tolist()

    return X, y, feature_names


def train_model(X, y):
    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # XGBoost baseline model
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist"
    )

    print("[*] Training XGBoost baseline model...")
    model.fit(X_train_scaled, y_train)

    # evaluation
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]

    print("\n🔎 VALIDATION METRICS (BASELINE XGBOOST)\n")
    print(classification_report(y_val, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_val, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        print("Could not compute ROC-AUC:", e)

    return model, scaler


def save_artifacts(model, scaler, feature_names):
    model_path = MODELS_DIR / "xgb_fraud_model.pkl"
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    features_path = MODELS_DIR / "feature_names.txt"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    print("\n✅ Saved model artifacts:")
    print(f"- Model:   {model_path}")
    print(f"- Scaler:  {scaler_path}")
    print(f"- Features:{features_path}")


def main():
    print("[*] Loading processed dataset...")
    df = load_data()

    print("[*] Preparing features and labels...")
    X, y, feature_names = prepare_xy(df)

    print(f"[*] Dataset size: {len(df)} rows, {X.shape[1]} features")

    model, scaler = train_model(X, y)

    save_artifacts(model, scaler, feature_names)

    print("\n[✓] Day 4 complete: baseline model trained and saved.")


if __name__ == "__main__":
    main()
