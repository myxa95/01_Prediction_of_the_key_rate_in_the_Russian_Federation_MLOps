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

    # Сохранение DataFrame df в файл data/df.csv
    output_df_path = config['preprocessing']['df_path']
    df.to_csv(output_df_path, index=False)

    return df

def load_data(
    data: str, type_data: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param data: данные
    :param type_data: тип датасет (train/test)
    :return: датасет, датасет в формате BytesIO
    """
    dataset = pd.read_csv(data)
    st.write("Dataset load")
    st.write(dataset.head())

    # Преобразовать dataframe в объект BytesIO (для последующего анализа в виде файла в FastAPI)
    dataset_bytes_obj = io.BytesIO()
    # запись в BytesIO буфер
    dataset.to_csv(dataset_bytes_obj, index=False)
    # Сбросить указатель, чтобы избежать ошибки с пустыми данными
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.csv", dataset_bytes_obj, "multipart/form-data")
    }
    return dataset, files
