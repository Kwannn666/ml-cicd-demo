FROM python:3.10-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS serve
WORKDIR /app
COPY app ./app
COPY artifacts ./artifacts
ENV LOCAL_MODEL_PATH=/app/artifacts/model.pkl
EXPOSE 8000
CMD ["uvicorn","app.serve:app","--host","0.0.0.0","--port","8000","--workers","1"]
