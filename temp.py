import cudf.pandas
cudf.pandas.install()

def download_file_if_not_exists(file_url, output_path):
    """Скачивает файл с Google Drive, если он ещё не существует в указанной директории."""
    # Проверка наличия файла
    if os.path.exists(output_path):
        print(f"Файл '{output_path}' уже существует.")
    else:
        print(f"Файл '{output_path}' не найден. Начинаю загрузку...")
        gdown.download(file_url, output_path, quiet=False)
        print(f"Файл '{output_path}' успешно загружен.")

# Указываем URL и путь к файлу
file_url = 'https://drive.google.com/uc?id=15pofNbomaoUap41Rcn1uNGeiJIqFd2qe'
output_file_name = 'wildberries_reviews.csv.gz'  # Укажите реальное имя файла, которое хотите сохранить
output_path = os.path.join(os.getcwd(), output_file_name)  # Полный путь к файлу

download_file_if_not_exists(file_url, output_path)

import subprocess

def install_spacy_model_if_not_exists(model_name):
    """Устанавливает модель spaCy, если она ещё не установлена."""
    try:
        # Получаем список установленных моделей
        result = subprocess.run(
            ['python', '-m', 'spacy', 'info'],
            capture_output=True, text=True, check=True
        )
        
        # Проверяем наличие модели в списке установленных
        if model_name in result.stdout:
            print(f"Модель {model_name} уже установлена.")
        else:
            print(f"Модель {model_name} не найдена. Устанавливаю...")
            subprocess.run(['python', '-m', 'spacy', 'download', model_name], check=True)
            print(f"Модель {model_name} успешно установлена.")
    
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении команды: {e}")

# Замените 'ru_core_news_lg' на нужное имя модели
install_spacy_model_if_not_exists('ru_core_news_lg')
df_raw = pd.read_csv('wildberries_reviews.csv.gz', compression='gzip')