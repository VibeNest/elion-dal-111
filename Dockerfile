FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Воспроизводимая установка из лока (pinned версии; grpcio-tools тоже в локе).
COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock

COPY proto ./proto
COPY scripts ./scripts
COPY src ./src

# Генерируем gRPC-код в src/elion_dal/grpc_gen, затем ставим сам пакет без зависимостей.
RUN python scripts/gen_proto.py && pip install --no-cache-dir --no-deps .

EXPOSE 50051 8080

CMD ["python", "-m", "elion_dal.service.server"]
