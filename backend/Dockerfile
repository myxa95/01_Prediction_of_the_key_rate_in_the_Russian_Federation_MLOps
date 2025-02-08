FROM python:3.11-slim-buster

WORKDIR /app/backend
COPY . .


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++

# Установка Poetry
RUN pip install poetry

# Установка зависимостей с помощью Poetry
RUN poetry install --no-root

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]