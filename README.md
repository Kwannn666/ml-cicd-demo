# ML CI/CD Demo (GitHub Actions → Train → MLflow → Dockerize → Serve)

This is a minimal, production-leaning template that:
- trains a simple model on push,
- logs metrics/artifacts to MLflow,
- builds a Docker image for inference (FastAPI),
- (optionally) pushes to GHCR and deploys via SSH.

> Works locally with Docker/Jupyter. CI runs on GitHub-hosted runner (CPU).

## Quick Start (Local)

1) **Create venv** (or use your Jupyter Python):
```bash
pip install -r requirements.txt
python train.py
```
This writes `artifacts/model.pkl` and prints a validation AUC.

2) **Build & run API**:
```bash
docker build -t ml-cicd-demo:local .
docker run -p 8000:8000 --rm ml-cicd-demo:local
```
Open http://localhost:8000/docs and try `/predict` with:
```json
{
  "rows": [{"f1":0.1,"f2":1.2,"f3":-0.2,"f4":3.3,"f5":0.5,"f6":1.1}]
}
```

## GitHub Actions CI/CD

1) Create a new GitHub repo and push this project.
2) Set **Secrets** (Repo → Settings → Secrets and variables → Actions):
   - `GHCR_TOKEN` (Personal Access Token with `packages:write`)
   - (Optional for deploy) `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`
   - (Optional) `MLFLOW_TRACKING_URI` if you have a remote MLflow server.
3) Edit `.github/workflows/cicd.yml`:
   - Replace `OWNER/REPO` with your GitHub org/repo.
4) Push to `main`. Workflow will:
   - run `train.py` → save `artifacts/`,
   - build & push image to GHCR,
   - (optional) SSH deploy with `docker-compose.prod.yml`.

## Customize
- Replace `data/train.csv` with your dataset and adjust `TARGET` env var in the workflow.
- Swap model to XGBoost/LightGBM in `train.py`.
- If you have MLflow Model Registry, set `USE_REGISTRY=1` and `MODEL_URI` in compose.

## Notes
- CI is CPU-only by default; local training can use GPU-friendly libs if you install them.
- This is a template: keep it simple, then iterate with tests, canary releases, etc.
