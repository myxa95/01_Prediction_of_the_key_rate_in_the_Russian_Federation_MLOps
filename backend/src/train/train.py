"""
Программа: Обучение модели Prophet на данных.
Функция оптимизации гиперпараметров модели Prophet.
Генерирует прогноз на заданное количество дней вперед.
Версия: 1.0
"""

import json
import logging
import joblib
import numpy as np
import optuna
from optuna.pruners import MedianPruner
import pandas as pd
import yaml

from prophet import Prophet
from prophet.diagnostics import cross_validation

CONFIG_PATH = '../config/params.yml'

with open(CONFIG_PATH, encoding='utf-8') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
data_path = config['train']['data_path']

# Чтение DataFrame df в файл data/df.csv
df_path = config['preprocessing']['df_path']
df = pd.read_csv(df_path)

# Чтение DataFrame df_train в файл data/df_train.csv
train_path = config['train']['train_path']
df_train = pd.read_csv(train_path)

# Чтение DataFrame df_test в файл data/df_test.csv
test_path = config['train']['test_path']
df_test = pd.read_csv(test_path)

def objective(trial, train_data: pd.DataFrame) -> float:
    """
    Целевая функция для оптимизации гиперпараметров Prophet с помощью Optuna.

    Параметры:
        trial (optuna.Trial): Объект пробного запуска Optuna
        train_data (pd.DataFrame): Данные для обучения

    Возвращает:
        float: Значение MAE для оценки качества модели
    """
    try:
        # Гиперпараметры для настройки
        params = {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10),
            "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10),
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
            "changepoint_range": trial.suggest_float("changepoint_range", 0.8, 0.95),
        }

        # модель Prophet с гиперпараметрами
        model = Prophet(**params)
        model.fit(train_data)

        # Кросс-валидация
        cv_results = cross_validation(model, initial="730 days", period="180 days", horizon="30 days")
        mae = np.mean(np.abs(cv_results["y"] - cv_results["yhat"]))

        return mae
    except Exception as e:
        logging.error("Error in objective function: %s", str(e))
        raise

def optimize_prophet_hyperparameters(train_data: pd.DataFrame, config):
    """
    Функция оптимизации гиперпараметров модели Prophet.

    Параметры:
    - train_data (pd.DataFrame): Данные для обучения модели.
    - config (dict): Словарь с конфигурацией.

    Возвращает:
    - prophet_best_params (dict): Лучшие параметры модели Prophet.
    """
    config_path = '../config/params.yml'
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Выполнение поиска гиперпараметров с помощью Optuna
    study = optuna.create_study(direction="minimize", pruner=MedianPruner())
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    study.optimize(lambda trial: objective(trial, train_data),
                   n_trials=config["train"]["N_TRIALS"],
                   timeout=config["train"]["TIMEOUT"],
                   )
    
    print("Модель и параметры сохранены")
    print("Лучшие параметры:", study.best_params)
    print("Лучшее значение:", study.best_value)

    return study

def train_model(data: pd.DataFrame, **kwargs):
    """
    Обучение модели Prophet на данных.

    Параметры:
        pandas.DataFrame: Данные для обучения.
        **kwargs: Параметры для модели Prophet.
    Возвращает:
        Обученная модель.
    """
    model = Prophet(**kwargs)
    model.fit(data)

    return model

def generate_forecast(model, pred_days):
    """
    Генерирует прогноз на заданное количество дней вперед.

    Параметры:
    - model: модель, используемая для прогнозирования
    - pred_days: количество дней, на которое нужно сделать прогноз

    Возвращает:
    - forecast: DataFrame с прогнозом
    """


    future = df_test[['ds']].copy()
    future = pd.concat([future, model.make_future_dataframe(periods=pred_days, freq="D")], ignore_index=True)
    forecast = model.predict(future)

    # Ограничение прогнозируемых значений
    # floor_value = 0 
    # forecast['yhat'] = forecast['yhat'].clip(lower=floor_value)

    return forecast

# # ДЛЯ ТЕСТА
# # Поиск оптимальных параметров
# study = optimize_prophet_hyperparameters(df_train, config)
# # Обучение на лучших параметрах
# reg_model = train_model(df=df_train, **study.best_params)

# # Период, который надо отрезать и предсказать (проверка модели)
# pred_days = int(df.shape[0]*config['parsing']['pred_days'])
# # Создание DataFrame с прогнозом
# df_forecast = generate_forecast(reg_model, pred_days)
# print('df_forecast_uncut:', df_forecast.shape)

# # Отбор только прогнозируемых значений
# df_forecast = df_forecast[-pred_days:]

# print('pred_days', pred_days)
# print('df:', df.shape)
# print('df_train+df_test:', len(df_test)+len(df_train))
# print('df_train:', df_train.shape)
# print('df_test:', df_test.shape)
# print('df_forecast:', df_forecast.shape)
# print(df_test.head(5))
# print(df_forecast.head(5))
# print(df_test.tail(5))
# print(df_forecast.tail(5))

