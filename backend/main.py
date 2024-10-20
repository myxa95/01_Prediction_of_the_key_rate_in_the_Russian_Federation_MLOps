"""
Модель прогнозирования ключевой ставки ЦБ РФ
Версия 1.0
"""

import warnings
import optuna

import uvicorn
from fastapi import FastAPI
from fastapi import File
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.data.get_metrics import get_metrics, save_metrics, load_metrics
from src.data.get_data import get_dataset

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

@app.post("/train")
def train():
    """
    Train model, logging metrics
    return: None
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(metrics_path=CONFIG_PATH)
    return {"message": "Model trained", "metrics": metrics}

if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        """Обработчик сигнала для корректного завершения программы."""
        print('Exiting gracefully...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        print(f"Error: {e}")


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
