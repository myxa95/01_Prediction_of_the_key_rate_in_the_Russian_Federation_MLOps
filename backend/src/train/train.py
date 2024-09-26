"""
Программа: Обучение модели Prophet на данных.
Функция оптимизации гиперпараметров модели Prophet.
Генерирует прогноз на заданное количество дней вперед.
Версия: 1.0
"""

import os
import json
import logging
import numpy as np
import optuna
import pandas as pd
import yaml

from prophet import Prophet
from prophet.diagnostics import cross_validation

CONFIG_PATH = '../../../config/params.yml'
with open(CONFIG_PATH, encoding='utf-8') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
data_path = config['train']['data_path']

# Чтение DataFrame df_train в файл data/df_train.csv
train_path = config['preprocessing']['train_path']
df_train = pd.read_csv(train_path)

# Чтение DataFrame df_test в файл data/df_test.csv
test_path = config['preprocessing']['test_path']
df_test = pd.read_csv(test_path)

def optimize_prophet_hyperparameters(df_train: pd.DataFrame, model_path, params_path, config):
    """
    Функция оптимизации гиперпараметров модели Prophet.

    Параметры:
    - df_train (pd.DataFrame): Данные для обучения модели.
    - model_path (str): Путь к директории, где будет сохранена лучшая модель.
    - params_path (str): Путь к директории, где будут сохранены лучшие параметры.
    - config (dict): Словарь с конфигурацией.

    Возвращает:
    - prophet_best_params (dict): Лучшие параметры модели Prophet.
    """

    # Определите целевую функцию для оптимизации
    def objective(trial):
        best_score = float("inf")
        # Гиперпараметры для настройки
        changepoint_prior_scale = trial.suggest_float(
            "changepoint_prior_scale", 0.001, 0.5
        )
        seasonality_prior_scale = trial.suggest_float(
            "seasonality_prior_scale", 0.01, 10
        )
        holidays_prior_scale = trial.suggest_float("holidays_prior_scale", 0.01, 10)
        seasonality_mode = trial.suggest_categorical(
            "seasonality_mode", ["additive", "multiplicative"]
        )

        # Создайте модель Prophet с гиперпараметрами
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
        )

        # Обучите модель
        model.fit(df_train)

        # Выполните кросс-валидацию
        cv_results = cross_validation(
            model, initial="730 days", period="180 days", horizon="30 days"
        )

        # Расчет MAE
        mae = np.mean(np.abs(cv_results["y"] - cv_results["yhat"]))

        score = mae

        # Обновляем best_score только если MAE лучше
        if score < best_score:
            best_score = score

        return score

    # Проверка, существует ли сохраненная модель и лучшие параметры
    best_model_file = os.path.join(model_path, "prophet_best_model.json")
    best_params_file = os.path.join(params_path, "prophet_best_params.json")
    prophet_best_params = None
    if os.path.exists(best_model_file) and os.path.exists(best_params_file):
        print("Модель и параметры уже сохранены.")
        with open(best_model_file, "r") as f:
            prophet_best_model = json.load(f)
        with open(best_params_file, "r") as f:
            prophet_best_params = json.load(f)
        print("Лучшие параметры:", prophet_best_params)
    else:
        print("Модель или параметры не сохранены, выполняем поиск гиперпараметров")
        # Выполнение поиска гиперпараметров с помощью Optuna
        study = optuna.create_study(direction="minimize")
        best_score = float("-inf")
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        study.optimize(
            objective,
            n_trials=config["train"]["N_TRIALS"],
            timeout=config["train"]["TIMEOUT"],
        )
        prophet_best_params = study.best_params
        # Сохранение модели и лучших параметров
        with open(best_model_file, "w") as f:
            json.dump(prophet_best_params, f, indent=4)
        with open(best_params_file, "w") as f:
            json.dump(study.best_params, f, indent=4)
        print("Модель и параметры сохранены")
        print("Лучшие параметры:", study.best_params)

    return prophet_best_params

def generate_forecast(model, pred_days):
    """
    Генерирует прогноз на заданное количество дней вперед.

    Параметры:
    - model: модель, используемая для прогнозирования
    - pred_days: количество дней, на которое нужно сделать прогноз

    Возвращает:
    - forecast: DataFrame с прогнозом
    """
    future = model.make_future_dataframe(periods=pred_days, freq="D")
    forecast = model.predict(future)
    return forecast


def train_model(df: pd.DataFrame, **kwargs):
    """
    Обучение модели Prophet на данных.

    Параметры:
        pandas.DataFrame: Данные для обучения.
        **kwargs: Параметры для модели Prophet.
    Возвращает:
        Обученная модель.
    """
    model = Prophet(**kwargs)
    model.fit(df)

    return model