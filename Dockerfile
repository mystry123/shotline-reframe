# CPU image for the reframe service. GPU later: swap onnxruntime for
# onnxruntime-gpu and use a CUDA base image; the code doesn't change.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY reframe ./reframe
COPY models ./models
ENV REFRAME_CACHE_DIR=/data/cache
EXPOSE 8090
USER nobody
CMD ["uvicorn", "reframe.service:app", "--host", "0.0.0.0", "--port", "8090"]
