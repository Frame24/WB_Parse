# data_processing.py
import pandas as pd

def load_data(filepath, nrows=None):
    """
    Загружает данные из CSV файла.

    :param filepath: Путь к CSV файлу.
    :param nrows: Количество строк для загрузки, если указано.
    :return: DataFrame с загруженными данными.
    """
    return pd.read_csv(filepath, nrows=nrows)
