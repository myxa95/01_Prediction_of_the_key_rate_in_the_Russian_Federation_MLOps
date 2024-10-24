import numpy as np
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, mean_squared_log_error
from datetime import datetime

def r2_adjusted(y_true: np.ndarray, y_pred: np.ndarray,
                X_test: np.ndarray) -> float:
    """Коэффициент детерминации (множественная регрессия)"""
    N_objects = len(y_true)
    N_features = X_test.shape[1]
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (N_objects - 1) / (N_objects - N_features - 1)


def mpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean percentage error"""
    return np.mean((y_true - y_pred) / y_true) * 100


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error"""
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percent Error"""
    return np.sum(np.abs(y_pred - y_true)) / np.sum(y_true) * 100


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.345):
    """Функция ошибки Хьюбера"""
    assert len(y_true) == len(y_pred), 'Разные размеры данных'
    huber_sum = 0
    for i in range(len(y_true)):
        if abs(y_true[i] - y_pred[i]) <= delta:
            huber_sum += 0.5 * (y_true[i] - y_pred[i])**2
        else:
            huber_sum += delta * (abs(y_true[i] - y_pred[i]) - 0.5 * delta)
    huber_sum /= len(y_true)
    return huber_sum


def logcosh(y_true: np.ndarray, y_pred: np.ndarray):
    """функция ошибки Лог-Кош"""
    return np.sum(np.log(np.cosh(y_true - y_pred)))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    The Root Mean Squared Log Error (RMSLE) metric 
    Логаритмическая ошибка средней квадратичной ошибки
    """
    try:
        return np.sqrt(mean_squared_log_error(y_true, y_pred))
    except:
        return None

def get_metrics(y_test: np.ndarray,
                y_pred: np.ndarray,
                name: str = None,):
    """Генерация таблицы с метриками для задачи регрессии"""
    df_metrics = pd.DataFrame()

    df_metrics['model'] = [name]

    df_metrics['MAE'] = mean_absolute_error(y_test, y_pred)
    df_metrics['MAPE_%'] = mean_absolute_percentage_error(y_test, y_pred)
    df_metrics['MSE'] = mean_squared_error(y_test, y_pred)
    df_metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))


    return df_metrics

def save_metrics(df_metrics: pd.DataFrame, model: object, metrics_path: str):
    """
    Сохранение метрик в файл
    """
    df_metrics['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_metrics.to_json(metrics_path, orient='records', lines=True, mode='a')
    with open(metrics_path, 'w', encoding='utf-8') as file:
        json.dump(df_metrics, file)

def load_metrics(metrics_path: str):
    """
    Загрузка метрик из файла
    """
    with open(metrics_path, 'r', encoding='utf-8') as file:
        df_metrics = json.load(file)
    return df_metrics

def get_dict_metrics(y_test: pd.DataFrame,
                     y_pred: pd.DataFrame,
                     name: str = None) -> dict:
    """Генерация словаря с метриками для задачи регрессии"""
    dict_metrics = {
        'model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE_%': mean_absolute_percentage_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    return dict_metrics


def save_dict_metrics(dict_metrics: dict, dict_metrics_path: str):
    """
    Сохранение метрик в файл как массив
    """
    # Загрузка существующих метрик, если файл существует
    try:
        with open(dict_metrics_path, 'r', encoding='utf-8') as file:
            existing_metrics = json.load(file)
    except FileNotFoundError:
        existing_metrics = []

    # Добавление новых метрик к существующим
    dict_metrics['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Добавление даты и времени
    existing_metrics.append(dict_metrics)

    # Сохранение обновленного массива метрик
    with open(dict_metrics_path, 'w', encoding='utf-8') as file:
        json.dump(existing_metrics, file)


def load_dict_metrics(dict_metrics_path: str) -> dict:
    """
    Загрузка метрик из файла
    """
    with open(dict_metrics_path, 'r', encoding='utf-8') as file:
        dict_metrics = json.load(file)
    return dict_metrics
