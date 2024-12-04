"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import yaml
import streamlit as st
from src.data.get_data import get_dataset
from src.data.interpolate_missing_values_and_prepare import interpolate_missing_values
from src.data.split_dataset import split_dataset
from src.plotting.get_plot import plot_key_rate, plot_features, plot_interpolate, plot_train_test_split
from src.plotting.create_features import create_features
from src.train.training import start_training

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
        plotting = plot_features(features)
        st.pyplot(plotting[0])

    if interpolate:
        st.markdown("Фильтрация выбросов при помощи IQR и интерполяция пропущенных значений")
        df_interpolated = data.copy()
        df_interpolated = interpolate_missing_values(df_interpolated, 'key_rate')
        plotting = plot_interpolate(data, df_interpolated)
        st.pyplot(plotting[0])
    
    if plot_train_test:
        st.markdown("График с разделением на train, test")
        df_split = data.copy()
        df_split = interpolate_missing_values(df_interpolated, 'key_rate')
        df_train, df_test = split_dataset(df_split, config)
        plotting = plot_train_test_split(df_train, df_test)
        st.pyplot(plotting[0])

def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model Prophet")
    # get params
    with open(CONFIG_PATH, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)

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
        "Tain model Prophet": training,

    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
