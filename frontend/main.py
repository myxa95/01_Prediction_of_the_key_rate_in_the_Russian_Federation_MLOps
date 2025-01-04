"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os
import yaml
import joblib
import streamlit as st
import pandas as pd
from src.data.get_data import get_dataset
from src.data.interpolate_missing_values_and_prepare import interpolate_missing_values
from src.data.split_dataset import split_dataset
from src.plotting.get_plot import plot_key_rate, plot_features, plot_interpolate, plot_train_test_split, plot_test_forecast
from src.plotting.create_features import create_features
from src.train.training import start_training, generate_forecast

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """

    st.image(
        "https://s0.rbk.ru/v6_top_pics/media/img/9/06/347286318883069.png",
        width=600,
    )

    st.markdown("# Описание проекта")
    st.title("Prediction of the key rate in the Russian Federation")
    st.write("Анализ и прогнозирование ключевой ставки ЦБ РФ")
    st.write("Проект использует данные по ключевой ставке Банка России для прогнозирования ее значений в будущем с помощью модели Prophet.")
    st.write("Основные шаги проекта:")
    st.write("- Сбор исторических данных по ключевой ставке ЦБ РФ с официального сайта.")
    st.write("- Предобработка данных: очистка, заполнение пропусков, преобразование в формат для модели.")
    st.write("- Разделение данных на обучающую и тестовую выборки.")
    st.write("- Обучение модели Prophet на обучающей выборке. Prophet использует аддитивную модель с трендом и сезонностью.")
    st.write("- Оценка качества модели на тестовой выборке по метрикам RMSE, MAE, MAPE.")
    st.write("- Использование лучшей модели для прогнозирования ключевой ставки на заданный период в будущем.")
    st.write("- Анализ полученных прогнозов, сравнение с фактическими данными и решениями ЦБ РФ по ставке")
    st.write("- Обучение модели Prophet на полной выборке и предсказание ставки на будущие периоды")

    # colums names
    st.markdown(
        """
        ### Описание полей 
            - ds - дата 
            - y - ключевая ставка
    """
    )

def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load dataset
    parsing_config = config['parsing']
    data = get_dataset(cfg=parsing_config)
    st.markdown("Последние курсы ставки рефинансирования ЦБ РФ:")
    st.write(data[:-5])

    # plotting with checkbox
    current_rate = st.sidebar.checkbox("Текущий курс ставки рефинансирования ЦБ РФ")
    features = st.sidebar.checkbox("Созданные признаки из df")
    interpolate = st.sidebar.checkbox("Фильтрация выбросов и интерполяция пропущенных значений")
    plot_train_test = st.sidebar.checkbox("График с разделением на train, test")

    if current_rate:
        st.markdown("График текущего курса и распределение ставки:")
        fig, ax = plot_key_rate(data)
        st.pyplot(fig)

    if features:
        st.markdown("Признаки по сезонам и дням недели:")
        features = create_features(data, col_datetime='date')
        fig, ax = plot_features(features)
        st.pyplot(fig)

    if interpolate:
        st.markdown("Фильтрация выбросов при помощи IQR и интерполяция пропущенных значений")
        df_interpolated = data.copy()
        df_interpolated = interpolate_missing_values(df_interpolated, 'key_rate')
        fig, ax = plot_interpolate(data, df_interpolated)
        st.pyplot(fig)
    
    if plot_train_test:
        st.markdown("График с разделением на train, test")
        df_split = data.copy()
        df_interpolated = interpolate_missing_values(df_split, 'key_rate')
        df_split = interpolate_missing_values(df_interpolated, 'key_rate')
        df_train, df_test = split_dataset(df_split, config)
        fig, ax = plot_train_test_split(df_train, df_test)
        st.pyplot(fig)

def training():
    """
    Тренировка модели
    """
    st.markdown("# Training test model Prophet")
    # get params
    with open(CONFIG_PATH, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train_test"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)

def forecast_test_model():
    """
    График прогноза и компоненты тест модели
    """
    st.markdown("# Plotting trained test model")
    # get params
    with open(CONFIG_PATH, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # load model
    model_path = config["train"]["model_path"]
    if os.path.exists(model_path):
        reg_model = joblib.load(model_path)
        st.success("Model loaded")
    else:
        st.error("Model not found")
        return  # Добавлено для выхода из функции, если модель не найдена

    # Чтение DataFrame df_test в файл data/df_test.csv
    test_path = config['train']['test_path']
    df_test = pd.read_csv(test_path)
    # Создание DataFrame с прогнозом
    df_forecast = generate_forecast(reg_model, config['train']['pred_days_forecast'], df_test)

    # График прогноза
    st.markdown("# График прогноза")
    model_course = reg_model.plot(df_forecast)
    st.pyplot(model_course)

    # Тренд, годовые и сезонные признаки
    st.markdown("# Тренд, годовые и сезонные признаки")
    model_trend = reg_model.plot_components(df_forecast)
    st.pyplot(model_trend)

    # График участка прогноза с фактическими данными
    st.markdown("# График участка прогноза с фактическими данными")
    fig = plot_test_forecast(df_test=df_test, df_forecast=df_forecast)
    st.pyplot(fig)

# def prediction():
#     """
#     Получение предсказаний путем ввода данных
#     """
#     st.markdown("# Prediction")
#     with open(CONFIG_PATH) as file:
#         config = yaml.load(file, Loader=yaml.FullLoader)
#     endpoint = config["endpoints"]["prediction_input"]
#     unique_data_path = config["preprocessing"]["unique_values_path"]

#     # проверка на наличие сохраненной модели
#     if os.path.exists(config["train"]["model_path"]):
#         evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
#     else:
#         st.error("Сначала обучите модель")

def main():
    """
    Сборка пайплайна 
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Tain test model Prophet": training,
        "Forecast key rate and plot test model": forecast_test_model,

    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
