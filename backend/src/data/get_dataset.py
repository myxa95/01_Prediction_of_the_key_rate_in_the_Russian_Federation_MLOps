import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date

def get_dataset(config):
    """
    Парсит ключевую ставку с URL сайта ЦБ РФ и возвращает pandas DataFrame.

    Параметры:
    config (dict): Словарь конфигурации, содержащий URL для парсинга.

    Возвращает:
    pd.DataFrame: DataFrame, содержащий спарсенные данные ключевой ставки.
    """
    url = config['parcing']["URL"] + date.today().strftime('%d.%m.%Y')
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find_all("table")

    df = pd.read_html(str(table))[0]
    df.iloc[:, 1:] /= 100
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True)
    df.columns = ['date', 'key_rate']

    return df