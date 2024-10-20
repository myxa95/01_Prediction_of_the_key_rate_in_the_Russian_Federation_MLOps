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

# def save_metrics(df_metrics: pd.DataFrame, model: object, metrics_path: str):
#     """
#     Сохранение метрик в файл
#     """
#     df_metrics['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     df_metrics.to_json(metrics_path, orient='records', lines=True, mode='a')
#     with open(metrics_path, 'w', encoding='utf-8') as file:
#         json.dump(df_metrics, file)

# def load_metrics(metrics_path: str):
#     """
#     Загрузка метрик из файла
#     """
#     with open(metrics_path, 'r', encoding='utf-8') as file:
#         df_metrics = json.load(file)
#     return df_metrics

def save_metrics(df_metrics: pd.DataFrame, model: object, metrics_path: str):
    """
    Сохранение метрик в файл
    """
    df_metrics['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Преобразование DataFrame в словарь для сериализации
    metrics_dict = df_metrics.to_dict(orient='records')
    
    # Загрузка существующих метрик, если файл существует
    existing_metrics = []
    try:
        with open(metrics_path, 'r', encoding='utf-8') as file:
            if file.readable():
                file.seek(0)  # Вернуться в начало файла
                content = file.read()
                if content:  # Проверка на пустоту
                    existing_metrics = json.loads(content)
    except FileNotFoundError:
        existing_metrics = []  # Если файл не найден, создаем пустой список

    # Объединение существующих и новых метрик
    all_metrics = existing_metrics + metrics_dict
    
    # Сохранение всех метрик в файл
    with open(metrics_path, 'w', encoding='utf-8') as file:
        json.dump(all_metrics, file)  # Сохранение в JSON

def load_metrics(metrics_path: str):
    """
    Загрузка метрик из файла
    """
    try:
        with open(metrics_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if content:  # Проверка на пустоту
                df_metrics = json.loads(content)
            else:
                df_metrics = []  # Если файл пуст, возвращаем пустой список
    except FileNotFoundError:
        df_metrics = []  # Если файл не найден, возвращаем пустой список
    except json.JSONDecodeError:
        df_metrics = []  # Если произошла ошибка декодирования, возвращаем пустой список

    return df_metrics