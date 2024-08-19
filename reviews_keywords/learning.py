import cudf.pandas  # Импортирование cuDF и активация его использования
cudf.pandas.install()  # Установка cuDF как основного интерфейса для pandas
import os
import pandas as pd
import os
import gdown
import os
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
from IPython.display import display
import numpy as np

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
# file_url = 'https://drive.google.com/uc?id=15pofNbomaoUap41Rcn1uNGeiJIqFd2qe'
file_url = 'https://drive.google.com/uc?id=1alondqI-2IHo__mYU7KQz4Ip8ytYGHXg'
output_file_name = 'wildberries_reviews.csv'  # Укажите реальное имя файла, которое хотите сохранить
output_path = os.path.join(os.getcwd(), output_file_name)  # Полный путь к файлу

download_file_if_not_exists(file_url, output_path)

# Путь к папке с CSV файлами
folder_path = './reviews_keywords/corrected_reviews'

# Получаем список всех файлов в папке
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Читаем и объединяем все CSV файлы в один датафрейм
df_list = [pd.read_csv(os.path.join(folder_path, file), index_col="id") for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=False)

combined_df.index = combined_df.index - 1
combined_df = pd.concat([pd.read_csv("wildberries_reviews.csv")[["corrected_text"]], combined_df], ignore_index=False)
# Выводим первые несколько строк объединенного датафрейма для проверки
combined_df.describe()

df_raw_big = pd.read_csv("wildberries_reviews.csv.gz", compression="gzip").drop("Unnamed: 0", axis=1)
df_raw_big.head()
result = combined_df.merge(df_raw_big, left_index=True, right_index=True, how='right')
result.describe()
df_raw_big = None
combined_df = None
result['corrected_text'] = result['corrected_text'].fillna(result['review_full_text'])
result.head()
# Оставляем только по 5 записей для каждого уникального значения в столбце 'product'
result_limited = result.groupby('product').head(10).reset_index(drop=True)
result_limited.describe()
import spacy
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели и токенайзера от Сбербанка
tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
model = AutoModel.from_pretrained('sberbank-ai/sbert_large_nlu_ru').to(device)

spacy.require_gpu()
# Загрузка и настройка модели SpaCy
nlp = spacy.load("ru_core_news_lg", disable=["ner", "tagger", "attribute_ruler", "lemmatizer"])

df = result_limited

# Преобразование pandas DataFrame в Hugging Face Dataset
dataset = Dataset.from_pandas(df)
import re

def clean_text(text):
    text = re.sub(r'[\n\r\t]+|\s{2,}', ' ', text)  # Объединяем шаги для замены пробелов
    text = re.sub(r'(?<!\.)\s*\.\s*|\s*\.\s*(?!\.)', '. ', text)  # Оптимизация замены точки
    return text.strip().rstrip('.')

def split_reviews_into_sentences(batch):
    # Очистка текстов
    cleaned_texts = [clean_text(text) for text in batch['corrected_text']]
    
    # Обработка текстов с помощью nlp.pipe с указанием batch_size
    docs = list(nlp.pipe(cleaned_texts, batch_size=64))  # Здесь 64 - пример значения

    # Извлечение предложений
    batch['sentences'] = [[sent.text for sent in doc.sents] for doc in docs]
    
    return batch

dataset = dataset.map(split_reviews_into_sentences, batched=True, batch_size=32)

# Преобразуем Dataset обратно в pandas DataFrame
df = dataset.to_pandas()

# Выполним explode по столбцу с предложениями
df_exploded = df.explode('sentences').reset_index(drop=True)

# Удаляем лишние столбцы, которые появились после explode
df_exploded = df_exploded.drop(columns=[col for col in df_exploded.columns if col.startswith('__index_level_')])

# Преобразуем DataFrame обратно в Hugging Face Dataset
dataset_exploded = Dataset.from_pandas(df_exploded)

from torch.cuda.amp import autocast

def compute_sentence_embeddings(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        with autocast():  # Используем mixed precision для ускорения
            outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# Функция для вычисления эмбеддингов для каждого предложения после explode
def compute_embeddings_after_explode(batch):
    sentences = batch['sentences']
    embeddings = compute_sentence_embeddings(sentences)
    batch['sentence_embeddings'] = embeddings
    return batch

# Применение функции
dataset = dataset_exploded.map(compute_embeddings_after_explode, batched=True, batch_size=128)
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm
import logging
from annoy import AnnoyIndex

# Отключение параллелизма в токенайзере Hugging Face
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Загрузка модели и токенайзера (без квантизации)
tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
model = AutoModel.from_pretrained('sberbank-ai/sbert_large_nlu_ru').to('cuda' if torch.cuda.is_available() else 'cpu')

# Настройка логирования
logging.basicConfig(filename='./reviews_keywords/clustering.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка модели spaCy для русского языка с использованием GPU
spacy.require_gpu()
nlp = spacy.load("ru_core_news_lg", disable=["parser", "ner"])

# Установка стоп-слов
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('russian'))

# Лемматизация с использованием pipe и батча
def lemmatize_sentences(sentences, batch_size=64):
    lemmatized_sentences = []
    for doc in nlp.pipe(sentences, batch_size=batch_size):
        lemmatized_sentences.append(" ".join([token.lemma_ for token in doc]))
    return lemmatized_sentences

# Функция для получения эмбеддинга предложения
def get_sentence_embedding(sentence):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).cpu().numpy()

# Вычисление эмбеддингов в пакетах
def compute_sentence_embeddings(sentences, batch_size=64):
    all_embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Вычисление эмбеддингов"):
        batch = sentences[i:i+batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.extend(embeddings)
    return np.vstack(all_embeddings)

# Создание и использование индекса Annoy для поиска ближайших соседей
def build_annoy_index(vectors, n_trees=50):  # Увеличили количество деревьев для улучшения точности
    f = vectors.shape[1]
    index = AnnoyIndex(f, 'angular')
    for i, vector in enumerate(vectors):
        index.add_item(i, vector)
    index.build(n_trees)
    return index

def query_annoy_index(index, vector, top_k=10):
    return index.get_nns_by_vector(vector, top_k)

# Основная функция для привязки предложений и классификации коротких предложений
def process_group(group_data):
    product_name, group, mask_embeddings, mask_words, threshold = group_data

    mask_index = build_annoy_index(mask_embeddings)

    all_sentences = group['sentences'].tolist()
    labeled_sentences = []

    for sentence in all_sentences:
        sentence_emb = get_sentence_embedding(sentence)
        indices = query_annoy_index(mask_index, sentence_emb.flatten(), top_k=1)
        max_similarity = cosine_similarity([sentence_emb.flatten()], [mask_embeddings[indices[0]]])[0][0]
        label = 1  # Default label for unassigned sentences

        if max_similarity > threshold:
            label = 0  # Assigned to a mask
        elif len(sentence.split()) in range(2, 5) and any(word in mask_words for word in sentence.split()):
            label = 2  # Short sentence classification

        labeled_sentences.append((product_name, sentence, label))

    return labeled_sentences

# Основной процесс с проверками и прогресс-баром
def process_reviews(df_exploded, mask_embeddings, mask_words, threshold=0.6):
    final_result = pd.DataFrame()

    # Обработка групп без параллелизма
    for group_data in tqdm(
        [(product_name, group, mask_embeddings, mask_words, threshold) for product_name, group in df_exploded.groupby('product')],
        desc="Обработка продуктов"
    ):
        results = process_group(group_data)
        final_result = pd.concat([final_result, pd.DataFrame(results, columns=['product', 'sentence', 'label'])], ignore_index=True)

    return final_result

# Пример вызова функций с кэшированием и пакетной обработкой
quality_phrases = [
    r'прекрасная вещь', r'замечательная вещь', 
    r'все пришло в идеальном состоянии', r'товар в отличном состоянии', 
    r'без повреждений', r'товар без дефектов', 
    r'все дошло целым', r'доставка без повреждений', r'идеальное состояние',
    r'очень доволен', r'очень довольна', r'товар понравился', r'качество понравилось'
]

functionality_phrases = [
    r'работает отлично', r'работает хорошо', r'всё работает'
]

gratitude_phrases = [
    r'спасибо', r'рекомендую', r'советую', r'продавец молодец', r'благодарен', r'благодарю', 
    r'советую к покупке', r'спасибо большое', r'всем советую', r'спасибо за товар', 
    r'спасибо продавцу', r'благодарю за товар', r'большое спасибо', r'очень благодарен', 
    r'спасибо за доставку', r'огромное спасибо', r'спасибо за качественный товар', 
    r'продавцу огромное спасибо', r'спасибо за оперативность', r'спасибо вам', 
    r'благодарен за товар', r'спасибо, всё хорошо', r'продавец молодец', r'спасибо за хорошее обслуживание',
    r'доволен сервисом', r'довольна сервисом'
]

delivery_phrases = [
    r'пришел быстро', r'быстрая доставка', r'пришел вовремя', r'заказ пришел целый и вовремя', 
    r'пришел целый', r'доставка вовремя', r'все пришло целым', r'товар пришел целым', r'пришел в срок',
    r'пришел вовремя и целым', r'получил заказ вовремя', r'доставка - во!', r'все пришло как надо', 
    r'пришел в полном порядке', r'все дошло целым', r'доволен доставкой', r'довольна доставкой'
]

confirmation_phrases = [
    r'всё соответствует', r'всё как в описании', r'всё как заявлено', r'соответствует описанию', 
    r'всё норм', r'всё хорошо', r'как всегда', r'без проблем', 
    r'нормально', r'всё норм', r'полностью доволен', r'полностью довольна', 
    r'всё понравилось'
]

simple_statements_phrases = [
    r'хорошая вещь', r'классная вещь', r'отличная вещь', r'удобно', r'нормально', r'работает', 
    r'работает отлично', r'работает хорошо', r'всё нормально', r'всё работает', r'всё ок', 
    r'всё окей', r'супер', r'класс', r'норм', r'отлично', r'хорошо', r'идеально', r'👍', r'👏', 
    r'😆', r'🔥', r'💯', r'класс', r'все супер', r'😊', r'доволен', r'довольна', 
    r'понравилось', "😊", "👍", "😍", "😂", "🛍️", "💯", "😆", "😁", "👏", "🔥",
    "🥰", "😎", "🤩", "❤️", "🤔", "🙌", "😜", "😉", "🤗", "😅",
    "👀", "🤷", "😋", "💖", "🌟", "😇", "😘", "🎉", "💪", "💥",
    "👌", "😄", "👋", "😏", "🙏", "🤝", "✨", "🤓", "🌸", "😌",
    "🥳", "🎁", "😑", "😳", "🙈", "😤", "👑", "😢", "🤤", "🤞"
]

# Определение масок и их эмбеддингов
mask_phrases = quality_phrases + functionality_phrases + gratitude_phrases + delivery_phrases + confirmation_phrases + simple_statements_phrases

# Лемматизация и создание эмбеддингов для масок
lemmatized_masks = lemmatize_sentences(mask_phrases)
mask_embeddings = compute_sentence_embeddings(lemmatized_masks)

# Создание списка всех лемматизированных слов из масок
all_words = []
for phrase in mask_phrases:
    all_words.extend(phrase.split())
mask_words = set(lemmatize_sentences(all_words))

# Вызов основной функции с эмбеддингами масок и логированием
final_result = process_reviews(df_exploded, mask_embeddings, mask_words)

final_result.to_csv("./reviews_keywords/final_result.csv")

# Показать результат
display(final_result[['product', 'sentence', 'label']])

import gc
gc.collect()

## Этап 2
import cudf.pandas  # Импортирование cuDF и активация его использования
cudf.pandas.install()  # Установка cuDF как основного интерфейса для pandas
import pandas as pd  # Импортирование pandas после установки cuDF


final_result.loc[final_result.label == 2, "label"] = 0
final_result

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
import logging
from transformers import TrainerCallback

# Разделение данных на тренировочную и валидационную выборки с использованием stratify
train_df, val_df = train_test_split(final_result, test_size=0.1, random_state=42, stratify=final_result['label'])

# Преобразование DataFrame в Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Загрузка токенизатора и модели
tokenizer = BertTokenizerFast.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
model = BertForSequenceClassification.from_pretrained('sberbank-ai/sbert_large_nlu_ru', num_labels=2).to('cuda')

# Токенизация данных
def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True, cache_file_name="train_data_cache")
val_dataset = val_dataset.map(tokenize_function, batched=True, cache_file_name="val_data_cache")

# Определение метрик
def compute_metrics(eval_pred, threshold=0.4):  # Устанавливаем порог ниже 0.5
    logits, labels = eval_pred
    predictions = (logits[:, 1] > threshold).astype(int)  # Применение порога
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
batch_train = 16
# Определение аргументов для тренировки
training_args = TrainingArguments(
    output_dir='./reviews_keywords/results',
    num_train_epochs=2,  # Уменьшение количества эпох
    per_device_train_batch_size=batch_train,
    per_device_eval_batch_size=batch_train * 2,
    warmup_steps=200,  # Уменьшение количества шагов прогрева
    weight_decay=0.01,
    logging_dir='./reviews_keywords/logs',
    logging_steps=5000 / batch_train,
    evaluation_strategy="steps",  # Валидация на каждом шаге
    eval_steps=20000 / batch_train,  # Валидация каждые 120 шагов
    fp16=True,  # Использование 16-битной точности
    gradient_accumulation_steps=2,  # Увеличение шага аккумуляции градиентов
)

# Настройка логирования
logging.basicConfig(filename='./reviews_keywords/clustering.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Создание коллбэка для дополнительного логирования
class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 100 == 0:  # Логирование каждые 100 шагов
            logging.info(f"Log at step {state.global_step}: {logs}")

# Trainer API от Hugging Face
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # Добавление метрик для оценки
    callbacks=[LogCallback()]  # Включение коллбэка для логирования
)

# Запуск обучения с валидацией
trainer.train()

# Сохранение дообученной модели
model.save_pretrained('./reviews_keywords/fine_tuned_model_10')
tokenizer.save_pretrained('./reviews_keywords/fine_tuned_model_10')
