import pandas as pd
import cupy as cp
import os
import language_tool_python
from spellchecker import SpellChecker
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import spacy
from textblob import TextBlob
import dask.dataframe as dd
from dask.distributed import Client, progress
from dask import delayed, compute
import gdown
import subprocess
import threading
import queue
from datetime import datetime

# Настройка логирования
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='/app/error_log.log', filemode='w')

def download_file_if_not_exists(file_url, output_path):
    """Скачивает файл с Google Drive, если он ещё не существует в указанной директории."""
    try:
        if os.path.exists(output_path):
            logging.info(f"Файл '{output_path}' уже существует.")
        else:
            logging.info(f"Файл '{output_path}' не найден. Начинаю загрузку...")
            gdown.download(file_url, output_path, quiet=False)
            logging.info(f"Файл '{output_path}' успешно загружен.")
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла: {e}")

file_url = 'https://drive.google.com/uc?id=15pofNbomaoUap41Rcn1uNGeiJIqFd2qe'
output_file_name = 'wildberries_reviews.csv.gz'
output_path = os.path.join(os.getcwd(), output_file_name)

download_file_if_not_exists(file_url, output_path)

def setup_languagetool():
    try:
        if not os.path.isdir(output):
            logging.info("Установка необходимых пакетов и загрузка LanguageTool...")
            os.system('apt update && apt install -y default-jre wget unzip')
            os.system('wget https://languagetool.org/download/LanguageTool-stable.zip && unzip -o LanguageTool-stable.zip')
        else:
            logging.info("LanguageTool уже установлен.")
    except Exception as e:
        logging.error(f"Ошибка при установке LanguageTool: {e}")

temp = language_tool_python.LanguageTool('ru-RU', config={'maxSpellingSuggestions': 1})
temp.close()

NUM_TOOLS = 6
MAX_CHECK_THREADS = 10
CHUNK_SIZE = 10000

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%d-%m-%Y_%H-%M-%S")
output_file_name = f"corrected_wildberries_reviews_{formatted_datetime}.csv"

def create_tool(tools_queue, pbar):
    try:
        tool = language_tool_python.LanguageTool('ru-RU', config={'maxSpellingSuggestions': 1, 'maxCheckThreads': MAX_CHECK_THREADS})
        tools_queue.put(tool)
        pbar.update(1)
    except Exception as e:
        logging.error(f"Ошибка при создании экземпляра LanguageTool: {e}")

def correct_text(text, tool, tools_queue):
    if not isinstance(text, str):
        return text
    try:
        corrected_text = tool.correct(text)
        return corrected_text if corrected_text != text else text
    except Exception as e:
        logging.error(f"Ошибка исправления текста: {text}\n{e}")
        # Попытка заменить инструмент
        try:
            logging.info("Замена инструмента LanguageTool новым экземпляром.")
            new_tool = language_tool_python.LanguageTool('ru-RU', config={'maxSpellingSuggestions': 1, 'maxCheckThreads': MAX_CHECK_THREADS})
            tools_queue.put(new_tool)
            corrected_text = new_tool.correct(text)
            return corrected_text if corrected_text != text else text
        except Exception as inner_e:
            logging.error(f"Ошибка при создании нового экземпляра LanguageTool: {inner_e}")
            return text

def process_text(text_queue, output_file_path, save_interval, lock, progress_bar, tools_queue):
    temp_results = []
    while not text_queue.empty():
        idx, text, tool = text_queue.get()
        corrected_text = correct_text(text, tool, tools_queue)
        if corrected_text != text:
            temp_results.append((idx, corrected_text))
        text_queue.task_done()
        progress_bar.update(1)

        if len(temp_results) >= save_interval:
            try:
                with lock:
                    save_partial_results(temp_results, output_file_path)
                    temp_results.clear()
            except Exception as e:
                logging.error(f"Ошибка при сохранении промежуточных результатов: {e}")

    if temp_results:
        try:
            with lock:
                save_partial_results(temp_results, output_file_path)
        except Exception as e:
            logging.error(f"Ошибка при сохранении оставшихся промежуточных результатов: {e}")

def save_partial_results(results, output_file_path):
    try:
        results.sort(key=lambda x: x[0])
        df = pd.DataFrame(results, columns=["id", 'corrected_text'])
        if os.path.exists(output_file_path):
            df.to_csv(output_file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file_path, index=False)
        logging.info(f"Сохранены промежуточные результаты в {output_file_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении результатов: {e}")

def process_data_chunk(chunk, output_file_path, tools, save_interval=1000, tools_queue=None):
    text_queue = queue.Queue()
    lock = threading.Lock()

    with tqdm(total=len(chunk), desc="Заполнение очереди строками") as pbar:
        for idx, row in chunk.iterrows():
            try:
                tool = tools[idx % len(tools)]
                text_queue.put((row.name, row['review_full_text'], tool))
                pbar.update(1)
            except Exception as e:
                logging.error(f"Ошибка при добавлении текста в очередь: {e}")

    with tqdm(total=text_queue.qsize(), desc="Обработка текста") as progress_bar:
        threads = []
        for _ in range(len(tools)):
            thread = threading.Thread(target=process_text, args=(text_queue, output_file_path, save_interval, lock, progress_bar, tools_queue))
            thread.start()
            threads.append(thread)

        text_queue.join()
        for thread in threads:
            thread.join()

def process_data(file_path, output_file_path, num_tools, save_interval=1000, chunk_size=10000, start_row=None, end_row=None):
    tools_queue = queue.Queue()
    threads = []

    with tqdm(total=num_tools, desc="Создание экземпляров LanguageTool") as pbar:
        for _ in range(num_tools):
            thread = threading.Thread(target=create_tool, args=(tools_queue, pbar))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    tools = [tools_queue.get() for _ in range(num_tools)]

    header = pd.read_csv(file_path, compression='gzip', nrows=0).columns.tolist()
    
    if start_row is not None and end_row is not None:
        total_rows = end_row - start_row
    else:
        total_rows = None

    for chunk in pd.read_csv(file_path, compression='gzip', skiprows=range(1, start_row+1), nrows=total_rows, chunksize=chunk_size, header=None, names=header):
        chunk.index += start_row
        process_data_chunk(chunk, output_file_path, tools, save_interval, tools_queue)

process_data('wildberries_reviews.csv.gz', f"/app/{output_file_name}", num_tools=NUM_TOOLS, save_interval=1000, chunk_size=CHUNK_SIZE, start_row=1409881, end_row=None)
