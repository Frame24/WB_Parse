import os
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import language_tool_python

# Функция инициализации language_tool_python
def init_tool():
    return language_tool_python.LanguageTool('ru-RU', config={'maxSpellingSuggestions': 1})

# Функция исправления текста
def correct_text(text, tool):
    if not isinstance(text, str):
        return text
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# Функция обработки партиций
def process_partition(partition, tool):
    partition['corrected_text'] = partition['review_full_text'].apply(lambda x: correct_text(x, tool))
    return partition

# Функция для обработки данных с использованием Dask
def process_data(file_path, n_rows):
    # Чтение данных и создание Dask DataFrame
    df_sample = pd.read_csv(file_path, compression='gzip', nrows=n_rows)
    ddf = dd.from_pandas(df_sample, npartitions=12)
    ddf = ddf.repartition(partition_size='100MB')

    # Инициализация инструмента для всех воркеров
    tool = init_tool()

    # Применение функции ко всем партициям
    ddf_corrected = ddf.map_partitions(lambda partition: process_partition(partition, tool))

    # Запись исправленных данных в CSV файл
    result = ddf_corrected.to_csv('wildberries_reviews_corrected.csv.gz', single_file=True, compression='gzip')

    # Ожидание завершения задачи
    from dask.distributed import progress
    progress(result)
    client.close()

# Настройка Dask LocalCluster с оптимальными параметрами
cluster = LocalCluster(
    n_workers=12,  # Устанавливаем количество воркеров
    threads_per_worker=1,  # Один поток на каждый воркер
    memory_limit='2GB',  # Устанавливаем лимит памяти на воркер
    dashboard_address=':8787'  # Включаем интерфейс мониторинга
)

client = Client(cluster)
print("Dask cluster is set up and connected.")
print("Monitor your Dask cluster at: http://localhost:8787/status")

# Применение функции обработки
process_data('wildberries_reviews.csv.gz', n_rows=1000)