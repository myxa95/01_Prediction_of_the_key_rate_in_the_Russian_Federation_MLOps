"""
Программа: Сборка и тренеровка модели Prophet
Версия: 1.0
"""

import os
import yaml
import joblib

from src.train.train import train_model, optimize_prophet_hyperparameters, generate_forecast
from src.data.get_data import get_dataset
from src.data.interpolate_missing_values_and_prepare import interpolate_missing_values
from src.data.prepare_data_for_prophet import prepare_data_for_prophet
from src.data.split_dataset import split_dataset
from src.data.get_metrics import get_dict_metrics, save_dict_metrics

def pipeline_training(config_path: str):
    """
    Получение данных с сайта ЦБ РФ,
    интерполяция пропущенных значений,
    разделение на обучающую и тестовую выборки,
    поиск лучших параметров с Optuna,
    тренировка модели Prophet,
    сохранение модели,
    сохранение параметров модели.

    Параметры:
        config_path (str): Путь к конфигурации.

    Возвращает:
        None
    """
    # Загрузка конфигурации
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    training_config = config['train']
    parsing_config = config['parsing']

    # Получение данных с сайта ЦБ РФ
    train_data = get_dataset(cfg=parsing_config)
    # Интерполяция пропущенных значений
    train_data = interpolate_missing_values(train_data, 'key_rate')
    # Подготовка данных для Prophet
    train_data = prepare_data_for_prophet(train_data)
    # Разделение на обучающую и тестовую выборки
    df_train, df_test = split_dataset(train_data, config)
    # Поиск оптимальных параметров
    study = optimize_prophet_hyperparameters(df_train, training_config)
    # Обучение на лучших параметрах
    reg_model = train_model(data=df_train, **study.best_params)
    # Создание DataFrame с прогнозом
    df_forecast = generate_forecast(reg_model, training_config['pred_days_forecast'])
    # Отбор только прогнозируемых значений
    df_forecast = df_forecast[-len(df_test):]
    # Получение метрик после обучения
    dict_metrics = get_dict_metrics(y_test=df_test['y'], y_pred=df_forecast['yhat'], name='Prophet_Optuna')
    # Сохранение метрик
    save_dict_metrics(dict_metrics, training_config['metrics_path'])
    # save_dict_metrics(dict_metrics, training_config['dict_metrics_path'])

    # Сохранение модели
    joblib.dump(reg_model, os.path.join(training_config["model_path"]))
    # Сохранение лучших параметров
    joblib.dump(study, os.path.join(training_config["params_path"]))

    # print(f"Загруженный объект: {type(study)}")
    # print(df_train.shape)
    # print(df_test.shape)
    # print(df_forecast.shape)    
    # print(df_train.head(5))
    # print(df_train.tail(5))
    # print(df_test.head(5))
    # print(df_test.tail(5))
    # print(df_forecast.head(5))
    # print(df_forecast.tail(5))

def pipeline_training_future(config_path: str):
    """
    Получение данных с сайта ЦБ РФ,
    интерполяция пропущенных значений,
    поиск лучших параметров с Optuna,
    тренировка модели Prophet,
    сохранение модели,
    сохранение параметров модели.

    Параметры:
        config_path (str): Путь к конфигурации.

    Возвращает:
        None
    """
    # Загрузка конфигурации
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    training_config = config['train']
    parsing_config = config['parsing']

    # Получение данных с сайта ЦБ РФ
    train_data = get_dataset(cfg=parsing_config)
    # Интерполяция пропущенных значений
    train_data = interpolate_missing_values(train_data, 'key_rate')
    # Подготовка данных для Prophet
    train_data = prepare_data_for_prophet(train_data)
    # Поиск оптимальных параметров
    study = optimize_prophet_hyperparameters(train_data, training_config)
    # Обучение на лучших параметрах
    reg_model = train_model(data=train_data, **study.best_params)
    # Создание DataFrame с прогнозом
    df_forecast = generate_forecast(reg_model, training_config['pred_days_forecast'])

    # Сохранение модели
    joblib.dump(reg_model, os.path.join(training_config["model_path_future"]))
    # Сохранение лучших параметров
    joblib.dump(study, os.path.join(training_config["params_path_future"]))