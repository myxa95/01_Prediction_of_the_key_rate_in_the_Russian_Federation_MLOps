"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os
import yaml
import streamlit as st


CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    st.markdown("#Прогнозирование ключевой ставки ЦБ РФ")
    st.image(config["frontend"]["main_image"], width=900)
    st.write(
        """
        Реализация интерфейса для получения маски дорог с изображения 
        со спутника. В данной работе ставилась задача обучить нейросеть
        находить дороги с изображения, по которой может проехать автомобиль.
        Тестировались модели Unet и DeepLabV3, графики метрик которых можно 
        найти во вкладке с анализом. Данные для обучения собирались вручную."""
    )


if __name__ == "__main__":
    main()
