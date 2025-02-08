FROM python:3.11-slim-buster

WORKDIR /app/frontend
COPY . .


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        gcc

# Установка Poetry
RUN pip install poetry

# Установка зависимостей с помощью Poetry
RUN poetry install --no-root


EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]