
"""
Программа: Разделение данных на обучающую и тестовую выборки и сохранение их в файлы.
Версия: 1.0
"""

import yaml

config_path = r'../config/params.yml'
config = yaml.load(open(config_path), Loader=yaml.FullLoader)
data_path = config['train']['data_path']

def split_dataset(df, config):
    """
    Разделение данных на обучающую и тестовую выборки и сохранение их в файлы.

    Параметры:
    df (pandas.DataFrame): Входные данные
    config (словарь): Словарь, содержащий параметры конфигурации
    data_path (строка): Путь к директории для сохранения файлов

    Возвращает:
    df_train (pandas.DataFrame): Обучающая выборка
    df_test (pandas.DataFrame): Тестовая выборка
    """
    pred_days = int(df.shape[0]*config['parcing']['pred_days'])
    df_train = df[:-pred_days]
    df_test = df[-pred_days:]
    
    # Сохранение DataFrame df_train в файл data/df_train.csv
    df_train.to_csv(f'{data_path}/df_train.csv', index=False)
    
    # Сохранение DataFrame df_test в файл data/df_test.csv
    df_test.to_csv(f'{data_path}/df_test.csv', index=False)
    
    return df_train, df_test