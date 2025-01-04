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

    return fig, ax


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
    sns.lineplot(x='date', y='key_rate', data=df_filtered, label='Интерполируемые данные', ax=ax)

    ax.set_xlabel('Год')
    ax.set_ylabel('Значение ставки')
    ax.set_title('Визуализация после вычисления выбросов')
    ax.legend(loc='best')
    ax.grid(True)

    return fig, ax


def plot_key_rate(df: pd.DataFrame):
    """
    Ви��уализация графика курса и графика распределения ключевой ставки ЦБ РФ.

    Параметры:
    df (pa.DataFrame): Данные о ключевой ставке.

    Возвращает:
    fig, ax: Объекты графиков для дальнейшего использования.
    """
    # Смотрим график курса ключевой ставки ЦБ РФ и график распределения
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    sns.set_theme(style="whitegrid", palette="Accent")

    sns.lineplot(x='date', y='key_rate', data=df, label='Ставка рефинансирования ЦБ РФ', ax=ax[0])
    ax[0].set_xlabel('График ставки рефинансирования РФ')
    ax[0].set_ylabel('Значение ставки')
    ax[0].legend(loc='best')
    ax[0].grid(True)

    sns.kdeplot(x=df['key_rate'], ax=ax[1], fill=True)
    ax[1].grid(True)
    ax[1].set_xlabel('График распределения ставки')
    ax[1].set_ylabel('Плотность вероятности')

    return fig, ax

def plot_train_test_split(df_train, df_test):
    """
    Отображает график с разделением train, test для наглядности.

    Параметры:
    df_train (pandas.DataFrame): Данные тренировочной выборки.
    df_test (pandas.DataFrame): Данные тестовой выборки.
    """
    fig, ax = plt.subplots(figsize = (10, 5))

    fig.set_figheight(5)
    fig.set_figwidth(15)
    df_train.set_index('date').plot(ax=ax, label= 'train', title='Визуальное разделение на тестовые и тренировочные данные')
    df_test.set_index('date').plot(ax=ax, label='test')
    ax.axvline(df_train['date'][-1:].values, ls='--', color='black')
    ax.legend(['Тренировочные данные', 'Тестовые данные'])
    ax.set_ylabel('Значение ставки')
    ax.set_xlabel('Год')
    plt.show()

    return fig, ax

def plot_forecast(model: any, forecast: any) -> None:
    """
    Отображает график прогноза и компоненты модели.

    Параметры:
    model (модель): Модель, использованная для прогнозирования.
    forecast (прогноз): Прогноз, полученный с помощью модели.

    Возвращает:
    None
    """
    # Смотрим график прогноза
    fig, ax = plt.subplots(1)
    model.plot(forecast, ax=ax)
    plt.show()

    # Смотрим тренд, годовые и сезонные признаки
    fig, ax = plt.subplots(1)
    model.plot_components(forecast)
    plt.show()

def plot_test_forecast(df_test: pd.DataFrame, df_forecast: pd.DataFrame) -> None:
    """
    Отображает график с данными тестовой выборки и прогнозом.

    Параметры:
    df_test (pandas.DataFrame): Данные тестовой выборки.
    df_forecast (pandas.DataFrame): Прогнозные данные.
    
    Возвращает:
    fig, ax: Объекты графиков для дальнейшего использования.
    """
    # Преобразование столбцов 'ds' в формат datetime
    df_test['ds'] = pd.to_datetime(df_test['ds'])
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])

    # Фильтрация df_forecast для получения последних значений, соответствующих df_test
    df_forecast = df_forecast.tail(len(df_test))

    fig, ax = plt.subplots()
    
    sns.lineplot(x='ds', y='y', data=df_test, label='Тестовые данные', ax=ax)
    sns.lineplot(x='ds', y='yhat', data=df_forecast, label='Прогноз', ax=ax)
    
    ax.set_xlabel('Дата')
    ax.set_ylabel('Значение ставки')
    ax.set_title('Сравнение тестовых данных и прогноза')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig

   
