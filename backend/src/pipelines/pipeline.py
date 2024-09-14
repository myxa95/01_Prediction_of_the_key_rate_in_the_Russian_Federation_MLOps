"""
Программа: Сборка и тренеровка модели Prophet
Версия: 1.0
"""

import os
import yaml

from ..data.get_dataset import get_dataset
from ..data.interpolate_missing_values_and_prepare import interpolate_missing_values
from ..data.split_dataset import split_dataset
from ..data.create_features import create_features
from ..train.train import train_model, optimize_prophet_hyperparameters

# Загрузка конфигурации
config_path = r'../config/params.yml'
with open(config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

train_config = config['train']

# Получение 
train_data = get_dataset()