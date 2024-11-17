"""
Модуль для создания новых признаков из столбца datetime в pandas DataFrame.
"""

import pandas as pd
from pandas import CategoricalDtype

# Определение категориальных типов данных для дней недели и месяцев
cat_day = CategoricalDtype(categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered= True)
cat_month = CategoricalDtype(categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ordered= True)

def create_features(data, col_datetime):
    """
    Создание новых признаков из столбца datetime в pandas DataFrame.

    Параметры:
    data (pd.DataFrame): Входной DataFrame
    col_datetime (str): Имя столбца datetime

    Возвращает:
    pd.DataFrame: DataFrame с добавленными новыми признаками
    """
    # Создание копии входного DataFrame, чтобы избежать его изменения
    data = data.copy()

    # Преобразование столбца datetime в формат datetime
    data[col_datetime] = pd.to_datetime(data[col_datetime])

    # Создание новых признаков
    data['weekday'] = data[col_datetime].dt.day_name().astype(cat_day)  # День недели
    data['month'] = data[col_datetime].dt.month_name().astype(cat_month)  # Месяц года
    data['year'] = data[col_datetime].dt.year  # Год
    data['quarter'] = data[col_datetime].dt.quarter  # Квартал года
    data['date_offset'] = (data[col_datetime].dt.month * 100 + data[col_datetime].dt.day - 320) % 1300  # Пользовательский признак смещения даты
    data['season'] = data[col_datetime].dt.month.map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'})  # Сезон года

    return data