import os, joblib, mlflow
from functools import lru_cache

USE_REGISTRY = os.getenv("USE_REGISTRY", "0") == "1"
MODEL_URI = os.getenv("MODEL_URI", "models:/demo_model/Production")
LOCAL_PATH = os.getenv("LOCAL_MODEL_PATH", "artifacts/model.pkl")

@lru_cache
def load_model():
    if USE_REGISTRY:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        return mlflow.pyfunc.load_model(MODEL_URI)
    return joblib.load(LOCAL_PATH)
