FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Воспроизводимая установка из лока (pinned версии; grpcio-tools тоже в локе).
COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock

# Тяжёлый стек рекомендуемого конфига st-bm25 (dense deepvk/USER-bge-m3 через sentence-transformers).
# torch ставим CPU-сборкой из официального индекса PyTorch — иначе с PyPI на Linux приедет
# CUDA-вариант (~5 ГБ). Версии зафиксированы под проверенное окружение. Это «тяжёлый» образ;
# лёгкий fastembed-вариант (e5-large, ONNX) собирается без этого слоя.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.12.0" \
 && pip install --no-cache-dir "sentence-transformers==5.5.1"

COPY proto ./proto
COPY scripts ./scripts
COPY src ./src

# Генерируем gRPC-код в src/elion_dal/grpc_gen, затем ставим сам пакет без зависимостей.
RUN python scripts/gen_proto.py && pip install --no-cache-dir --no-deps .

EXPOSE 8080

CMD ["python", "-m", "elion_dal.service.server"]
