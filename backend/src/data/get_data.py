"""
Программа: Парсит ключевую ставку с URL сайта ЦБ РФ и возвращает pandas DataFrame.
Версия: 1.0
"""

from io import BytesIO
import io
from datetime import date
from typing import Tuple, Dict
import requests
import pandas as pd
import yaml
import streamlit as st
from bs4 import BeautifulSoup



CONFIG_PATH = '../config/params.yml'
with open(CONFIG_PATH, encoding='utf-8') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

def get_dataset(cfg):
    """
    Парсит ключевую ставку с URL сайта ЦБ РФ и возвращает pandas DataFrame.

    Параметры:
    config (dict): Словарь конфигурации, содержащий URL для парсинга.

    Возвращает:
    pd.DataFrame: DataFrame, содержащий спарсенные данные ключевой ставки.
    """
    url = cfg["URL"] + date.today().strftime('%d.%m.%Y')
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Ошибка таймаута: не удалось получить данные с URL")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find_all("table")

    df = pd.read_html(str(table))[0]
    df.iloc[:, 1:] /= 100
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True)
    df.columns = ['date', 'key_rate']
    df = df.sort_values(by='date').reset_index(drop=True)

    # Сохранение DataFrame df в файл data/df.csv
    output_df_path = config['preprocessing']['df_path']
    df.to_csv(output_df_path, index=False)

    return df

def load_data(dataset_path: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    
    Параметры:
    dataset_path: Путь до данных

    Возвращает:
    pd.DataFrame: Датасет
    """

    return pd.read_csv(dataset_path)


# # ДЛЯ ТЕСТА
# parsing_config = config['parsing']
# dataset = get_dataset(cfg=parsing_config)
# print(dataset)