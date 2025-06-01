FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml .
COPY main.py .

RUN uv sync

CMD ["uv", "run", "main.py"]

