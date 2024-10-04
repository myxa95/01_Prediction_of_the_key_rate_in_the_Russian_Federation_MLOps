"""
Программа: Подготовка интеполируемых данных для Prophet.
Версия: 1.0
"""

import pandas as pd
import yaml

CONFIG_PATH = '../config/params.yml'
with open(CONFIG_PATH, encoding='utf-8') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

def prepare_data_for_prophet(df: pd.DataFrame):
    """
    Подготовка интеполируемых данных для Prophet путем переименования столбцов и сортировке по дате.

    Параметры:
    df (pd.DataFrame): Интеполируемые данные

    Возвращает:
    pd.DataFrame: Подготовленные данные для Prophet
    """

    # Переименовываем столбцы для Prophet
    df.columns = ['ds', 'y']

    # Сортируем данные по дате в порядке возрастания
    df = df.sort_values('ds')
    df = df.reset_index(drop=True)

    # Сохранение DataFrame df в файл data/df.csv
    df_path = config['preprocessing']['df_path']
    df.to_csv(df_path, index=False)
    
    return df