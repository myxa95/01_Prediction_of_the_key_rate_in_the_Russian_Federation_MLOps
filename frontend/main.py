"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os
import yaml
import streamlit as st
from src.data.get_data import load_data, get_dataset

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
    st.write(
        """Анализ и прогнозирование ключевой ставки ЦБ РФ"""
    )

    # Имена колонок
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

    # load and write dataset
    data = load_data(dataset_path=config["preprocessing"]["df_path"])
    st.write(data.head())


def main():
    """
    Сборка пайплайна 
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,

    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
