"""
Программа: Построение графика с признаками и визуализация оригинальных и отфильтрованных данных ключевой ставки
Версия: 1.0
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_features(data: pd.DataFrame):
    """
    Создает два графика для анализа ключевой ставки:
    1. Бары максимальных значений ставки по годам.
    2. Боксплот распределения ставки по дням недели и сезонам.

    Параметры:
    data (pd.DataFrame): DataFrame, содержащий данные о ключевой ставке, днях недели, сезоны.

    Возвращает:
    Два графика
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    year_group = pd.DataFrame(data.groupby('year')['key_rate'].max()).reset_index().sort_values('key_rate')

    sns.barplot(data=year_group, x='year', y='key_rate', ax=ax[0])
    ax[0].set_xlabel('Год')
    ax[0].set_ylabel('Значение ставки')

    sns.boxplot(data=data, x='weekday', y='key_rate', hue='season', ax=ax[1], linewidth=2)
    ax[1].set_xlabel('День недели')
    ax[1].set_ylabel('Значение ставки')

    return plt.show()


def plot_interpolate(df: pd.DataFrame, df_filtered: pd.DataFrame):
    """
    Визуализация оригинальных и отфильтрованных данных ключевой ставки.

    Параметры:
    df (pandas.DataFrame): Оригинальные данные ключевой ставки.
    df_filtered (pandas.DataFrame): Отфильтрованные данные ключевой ставки.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.set_theme(style="whitegrid", palette="Accent")

    sns.lineplot(x='date', y='key_rate', data=df, label='Оригинальные данные', ax=ax)
    sns.lineplot(x='date', y='key_rate', data=df_filtered, label='Отфильтрованные данные', ax=ax)

    ax.set_xlabel('Год')
    ax.set_ylabel('Значение ставки')
    ax.set_title('Визуализация после вычисления выбросов')
    ax.legend(loc='best')
    ax.grid(True)

    return plt.show()
