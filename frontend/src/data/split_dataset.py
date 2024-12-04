
"""
Программа: Разделение данных на обучающую и тестовую выборки и сохранение их в файлы.
Версия: 1.0
"""

import yaml
import pandas as pd

CONFIG_PATH = '../config/params.yml'
with open(CONFIG_PATH, encoding='utf-8') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
data_path = config['train']['data_path']

def split_dataset(df: pd.DataFrame, config):
    """
    Разделение данных на обучающую и тестовую выборки и сохранение их в файлы.

    Параметры:
    df (pd.DataFrame): Входные данные
    config (словарь): Словарь, содержащий параметры конфигурации
    train_path (строка): Путь к директории для сохранения файла
    test_path (строка): Путь к директории для сохранения файла

    Возвращает:
    df_train (pd.DataFrame): Обучающая выборка
    df_test (pd.DataFrame): Тестовая выборка
    """
    df = df.sort_values(by=df.columns[0], ascending=True)
    pred_days = int(df.shape[0]*config['parsing']['pred_days'])
    df_train = df[:-pred_days]
    df_test = df[-pred_days:]

    return df_train, df_test