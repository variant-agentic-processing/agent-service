FROM python:3.11-slim-bookworm
WORKDIR /app
RUN pip install poetry
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && poetry install --only main --no-root
COPY src/ ./src/
ENV PORT=8080
CMD ["python3", "-m", "src.main"]
