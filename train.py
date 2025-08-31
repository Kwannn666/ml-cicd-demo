import os, json
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

DATA = os.getenv("DATA_CSV", "data/train.csv")
TARGET = os.getenv("TARGET", "target")
ARTS = "artifacts"; os.makedirs(ARTS, exist_ok=True)

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "demo"))

    df = pd.read_csv(DATA)
    X = df.drop(columns=[TARGET]); y = df[TARGET]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run() as run:
        params = {"n_estimators": 300, "max_depth": None, "random_state": 42}
        model = RandomForestClassifier(**params).fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:,1]
        auc = roc_auc_score(yva, p)

        mlflow.log_params(params)
        mlflow.log_metric("val_auc", auc)
        mlflow.sklearn.log_model(model, "model")

        import joblib
        out = os.path.join(ARTS, "model.pkl")
        joblib.dump(model, out)
        with open(os.path.join(ARTS, "model_meta.json"), "w") as f:
            json.dump({"run_id": run.info.run_id, "val_auc": float(auc)}, f)

        print(f"[OK] val_auc={auc:.5f}, run_id={run.info.run_id}")

if __name__ == "__main__":
    main()
