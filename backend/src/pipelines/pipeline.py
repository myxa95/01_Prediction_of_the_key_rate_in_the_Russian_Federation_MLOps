"""
Программа: Сборка и тренеровка модели Prophet
Версия: 1.0
"""

import os
import yaml

from ..data.get_data import get_dataset
from ..data.interpolate_missing_values_and_prepare import interpolate_missing_values
from ..data.split_dataset import split_dataset
from ..data.create_features import create_features
from ..train.train import train_model, optimize_prophet_hyperparameters

# Загрузка конфигурации
# config_path = r'../config/params.yml'
# with open(config_path) as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)

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
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']
    training_config = config['training']
    parcing_config = config['parcing']

    # Получение данных с сайта ЦБ РФ
    train_data = get_dataset(url=parcing_config["URL"])

    # Интерполяция пропущенных значений
    train_data = interpolate_missing_values(train_data, 'key_rate')

    # Разделение на обучающую и тестовую выборки
    df_train, df_test = split_dataset(train_data, config)
    
    # Поиск оптимальных параметров
    study = optimize_prophet_hyperparameters(df_train, **training_config)

    # Обучение на лучших параметрах
    reg = train_model(df=df_train)