from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from .model_io import load_model

app = FastAPI(title="ML CI/CD Demo")

class PredictIn(BaseModel):
    rows: list[dict]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(inp: PredictIn):
    X = pd.DataFrame(inp.rows)
    model = load_model()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:,1].tolist()
        return {"probs": probs}
    preds = model.predict(X).tolist()
    return {"preds": preds}
