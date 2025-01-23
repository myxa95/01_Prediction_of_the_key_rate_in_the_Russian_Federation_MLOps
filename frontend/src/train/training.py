"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import yaml
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_timeline
import optuna
import pandas as pd

CONFIG_PATH = "../config/params.yml"

def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # Last metrics
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"], encoding='utf-8') as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {"MAE": 0, "MAPE_%": 0, "MSE": 0, "RMSE": 0}

    # Train
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Model trained!")

    new_metrics = output.json()["metrics"]

    # diff metrics
    mae, mape, mse, rmse = st.columns(4)

    mae.metric(
        "MAE",
        new_metrics["MAE"],
        f"{new_metrics['MAE']-old_metrics['MAE']:.3f}",
    )
    mape.metric(
        "MAPE_%",
        new_metrics["MAPE_%"],
        f"{new_metrics['MAPE_%']-old_metrics['MAPE_%']:.3f}",
    )
    mse.metric(
        "MSE",
        new_metrics["MSE"],
        f"{new_metrics['MSE']-old_metrics['MSE']:.3f}",
    )
    rmse.metric(
        "RMSE",
        new_metrics["RMSE"],
        f"{new_metrics['RMSE']-old_metrics['RMSE']:.3f}",
    )

    # plot study
    study = joblib.load(os.path.join(config["train"]["params_path"]))
    print(f"Загруженный объект: {type(study)}")
    if not isinstance(study, optuna.study.Study):
        raise ValueError("Загруженный объект не является экземпляром optuna.study.Study")

    # check ntrials
    if len(study.trials) > 1:
        # Визуализация важности параметров
        fig_imp = plot_param_importances(study)
        # Визуализация истории оптимизации
        fig_history = plot_optimization_history(study)
        # Визуализация временной шкалы
        fig_timeline = plot_timeline(study)

        # Отображение графиков в Streamlit
        st.plotly_chart(fig_imp, use_container_width=True)
        st.plotly_chart(fig_history, use_container_width=True)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.warning("Недостаточно пробных запусков для оценки важности параметров.")

def generate_forecast(model, pred_days: pd.DataFrame, df_test: pd.DataFrame):
    """
    Генерирует прогноз на заданное количество дней вперед.

    Параметры:
    - model: модель, используемая для прогнозирования
    - pred_days: количество дней, на которое нужно сделать прогноз

    Возвращает:
    - forecast: DataFrame с прогнозом
    """


    with open(CONFIG_PATH, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Чтение DataFrame df_test в файл data/df_test.csv
    test_path = config['train']['test_path']
    df_test = pd.read_csv(test_path)

    future = df_test[['ds']].copy() # копия дат из df_test
    future = pd.concat([future, model.make_future_dataframe(periods=pred_days, freq="D")], ignore_index=True)
    forecast = model.predict(future)

    # Сохранение DataFrame df в файл data/df_forecast.csv
    output_df_path = config['train']['df_forecast']
    forecast.to_csv(output_df_path, index=False)

    return forecast

def start_training_future(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """

    # Train
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Model trained!")

    # plot study
    study = joblib.load(os.path.join(config["train"]["params_path_future"]))
    print(f"Загруженный объект: {type(study)}")
    if not isinstance(study, optuna.study.Study):
        raise ValueError("Загруженный объект не является экземпляром optuna.study.Study")

    # check ntrials
    if len(study.trials) > 1:
        # Визуализация важности параметров
        fig_imp = plot_param_importances(study)
        # Визуализация истории оптимизации
        fig_history = plot_optimization_history(study)
        # Визуализация временной шкалы
        fig_timeline = plot_timeline(study)

        # Отображение графиков в Streamlit
        st.plotly_chart(fig_imp, use_container_width=True)
        st.plotly_chart(fig_history, use_container_width=True)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.warning("Недостаточно пробных запусков для оценки важности параметров.")


def generate_forecast_future(model, pred_days: pd.DataFrame, df: pd.DataFrame):
    """
    Генерирует прогноз на заданное количество дней вперед.

    Параметры:
    - model: модель, используемая для прогнозирования
    - pred_days: количество дней, на которое нужно сделать прогноз

    Возвращает:
    - forecast: DataFrame с прогнозом
    """


    with open(CONFIG_PATH, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Чтение DataFrame df в файл data/df.csv
    df_path = config['train']['df_path']
    df = pd.read_csv(df_path)

    future = df[['ds']].copy()  # копия дат из df
    future = pd.concat(
        [future, model.make_future_dataframe(periods=pred_days, freq="D")],
        ignore_index=True
    )
    forecast = model.predict(future)

    # Сохранение DataFrame df в файл data/df_forecast.csv
    output_df_path = config['train']['df_forecast_future']
    forecast.to_csv(output_df_path, index=False)

    return forecast
