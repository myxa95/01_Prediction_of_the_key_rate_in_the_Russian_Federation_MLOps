
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

# Чтение DataFrame df в файл data/df.csv
df_path = config['preprocessing']['df_path']
df = pd.read_csv(df_path)

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
    pred_days = int(df.shape[0]*config['parsing']['pred_days'])
    df_train = df[:-pred_days]
    df_test = df[-pred_days:]
    
    # Сохранение DataFrame df_train в файл data/df_train.csv
    train_path = config['preprocessing']['train_path']
    df_train.to_csv(train_path, index=False)
    
    # Сохранение DataFrame df_test в файл data/df_test.csv
    test_path = config['preprocessing']['test_path']
    df_test.to_csv(test_path, index=False)

    return df_train, df_test