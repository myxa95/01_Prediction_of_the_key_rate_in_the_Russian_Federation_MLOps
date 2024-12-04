"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history


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
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
