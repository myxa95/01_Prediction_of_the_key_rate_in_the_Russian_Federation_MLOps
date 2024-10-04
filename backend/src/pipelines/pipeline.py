"""
Программа: Сборка и тренеровка модели Prophet
Версия: 1.0
"""

import os
import json
import yaml

from ..train.train import train_model, optimize_prophet_hyperparameters, generate_forecast
from ..data.get_data import get_dataset
from ..data.interpolate_missing_values_and_prepare import interpolate_missing_values
from ..data.prepare_data_for_prophet import prepare_data_for_prophet
from ..data.split_dataset import split_dataset

def pipeline_training(config_path: str):
    """
    Получение данных с сайта ЦБ РФ,
    интерполяция пропущенных значений,
    разделение на обучающую и тестовую выборки,
    создание признаков,
    тренировка модели Prophet.

    Параметры:
        config_path (str): Путь к конфигурации.

    Возвращает:
        None
    """
    # Загрузка конфигурации
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config['preprocessing']
    training_config = config['train']
    parcing_config = config['parsing']

    # Получение данных с сайта ЦБ РФ
    train_data = get_dataset(cfg=parcing_config["URL"])
    # Интерполяция пропущенных значений
    train_data = interpolate_missing_values(train_data, 'key_rate')
    # Подготовка данных для Prophet
    train_data = prepare_data_for_prophet(train_data)
    # Разделение на обучающую и тестовую выборки
    df_train, df_test = split_dataset(train_data, config)
    # Поиск оптимальных параметров
    study = optimize_prophet_hyperparameters(df_train, **training_config)
    # Обучение на лучших параметрах
    reg = train_model(df=df_train, **study)
    # Создание DataFrame с прогнозом
    df_forecast = generate_forecast(reg, training_config['pred_days_forecast'])

    # Сохранение модели и параметров
    with open(training_config['model_path'], 'w', encoding='utf-8') as model_file:
        json.dump(reg, model_file)

    with open(training_config['params_path'], 'w', encoding='utf-8') as params_file:
        json.dump(study, params_file)
