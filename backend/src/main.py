"""
Модель прогнозирования ключевой ставки ЦБ РФ
Версия 1.0
"""

import warnings
import optuna
import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# переменная для запуска приложения
app = FastAPI()
CONFIG_PATH = '../config/params.yml'

@app.get("/Hello")
def welcome():
    """
    Hello!
    return: None
    """
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)