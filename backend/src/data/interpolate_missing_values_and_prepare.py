"""
Программа: Вычисление межквартильного размаха (IQR) и интерполяция пропущенных значений в указанном столбце.
Версия: 1.0
"""

import pandas as pd
import numpy as np
import yaml

CONFIG_PATH = '../../../config/params.yml'
with open(CONFIG_PATH, encoding='utf-8') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
data_path = config['train']['data_path']

# Чтение DataFrame df в файл data/df.csv
df_path = config['preprocessing']['df_path']
df = pd.read_csv(df_path)

def interpolate_missing_values(df: pd.DataFrame, column_name):
    """
    Фильтрация выбросов и интерполяция пропущенных значений в указанном столбце.

    Параметры:
    - df (pd.DataFrame): входные данные
    - column_name (str): имя столбца для обработки

    Возвращает:
    - df_filtered (pd.DataFrame): Интерполированные данные
    """

    # Определение последней даты, для заполнения графика интерполяцией
    last_date = df['date'].max()

    # Настройка фильтрации данных до последней даты
    mask = df['date'] < last_date

    # Вычисление межквартильного размаха (IQR)
    q1, q3 = df.loc[mask, column_name].quantile([0.25, 0.75])
    iqr = q3 - q1

    # Определение границ для выбросов
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Замена выбросов на NaN
    df_filtered = df.copy()
    df_filtered.loc[(df_filtered[column_name] < lower_bound) & mask, column_name] = np.nan
    df_filtered.loc[(df_filtered[column_name] > upper_bound) & mask, column_name] = np.nan

    # Интерполируем пропущенные значения
    df_filtered[column_name] = df_filtered[column_name].interpolate(method='nearest', order=3)

    return df_filtered

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
    output_df_path = config['preprocessing']['df_path']
    df.to_csv(output_df_path, index=False)

    return df
