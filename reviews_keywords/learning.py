import cudf.pandas  # Импортирование cuDF и активация его использования
cudf.pandas.install()  # Установка cuDF как основного интерфейса для pandas
import os
import pandas as pd
import os
import gdown
from IPython.display import display
import torch
import pyarrow.parquet as pq
import dask.dataframe as dd
# Очистка неиспользуемой памяти перед началом вычислений
torch.cuda.empty_cache()
import spacy
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter
from torch.cuda.amp import autocast
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
import re
import gc

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели и токенайзера от Сбербанка
tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
model = AutoModel.from_pretrained('sberbank-ai/sbert_large_nlu_ru').to(device)

spacy.require_gpu()
# Отключение параллелизма в токенайзере Hugging Face
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Настройка логирования
logging.basicConfig(filename='./reviews_keywords/clustering.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8')

# Загрузка модели spaCy для русского языка с использованием GPU
# spacy.require_gpu()
nlp = spacy.load("ru_core_news_lg", disable=["parser", "ner"])

# Установка стоп-слов
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('russian'))


# result = pd.read_csv("./reviews_keywords/wildberries_reviews_corrected.csv")
# result.info()

# # Оставляем только по 5 записей для каждого уникального значения в столбце 'product'
# result_limited = result.groupby('product').head(10000).reset_index(drop=True)
# result_limited.describe()

# df = result_limited

chunk_size = 50000  # количество строк для одного чанка
# Открываем файл для записи обработанных данныхimport os

# Указание пути к директории и файлу
directory = './reviews_keywords/automarkup/'
file_path = os.path.join(directory, 'final_result.csv')

# Создание директории, если она не существует
os.makedirs(directory, exist_ok=True)
with open('./reviews_keywords/automarkup/final_result.csv', 'w+') as f:
    for df in tqdm(pd.read_csv("./reviews_keywords/wildberries_reviews_corrected.csv", chunksize=chunk_size), desc="Processing chunks"):
        # Преобразование pandas DataFrame в Hugging Face Dataset
        spacy.require_gpu()
        # Загрузка и настройка модели SpaCy
        nlp = spacy.load("ru_core_news_lg", disable=["ner", "tagger", "attribute_ruler", "lemmatizer"])
        
        dataset = Dataset.from_pandas(df)

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


        def compute_sentence_embeddings(sentences):
            inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                with autocast():  # Используем mixed precision для ускорения
                    outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


        # Функция для вычисления эмбеддингов для каждого предложения после explode
        def compute_embeddings_after_explode(batch):
            sentences = batch['sentences']
            try:
                embeddings = compute_sentence_embeddings(sentences)
            except Exception as e:
                print(f"Ошибка при обработке: {e}")
                # Вы можете заменить неудачные эмбеддинги на нулевые или другие значения по умолчанию
                embeddings = torch.zeros((len(sentences), 768))  # Замените 768 на размерность эмбеддинга, если она другая
            batch['sentence_embeddings'] = embeddings
            return batch

        # Очистка неиспользуемой памяти перед началом вычислений
        torch.cuda.empty_cache()

        spacy.require_gpu()
        nlp = spacy.load("ru_core_news_lg", disable=["parser", "ner"])

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
            for i in range(0, len(sentences), batch_size):
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

                unique_words = len(set(sentence.split()))
                
                if max_similarity > threshold:
                    label = 0  # Assigned to a mask
                elif len(sentence.split()) in range(3, 5) and any(word in mask_words for word in sentence.split()):
                    label = 2  # Short sentence classification
                elif len(sentence.split()) in range(1, 2) and any(word in mask_words for word in sentence.split()):
                    label = 2  # Short sentence classification
                elif threshold - max_similarity < 0.1:
                    label = -1 

                labeled_sentences.append((product_name, sentence, label, max_similarity))

            return labeled_sentences

        # Основной процесс с проверками и прогресс-баром
        def process_reviews(df_exploded, mask_embeddings, mask_words, threshold=0.65):
            final_result = pd.DataFrame()

            # Обработка групп без параллелизма
            for group_data in tqdm(
                [(product_name, group, mask_embeddings, mask_words, threshold) for product_name, group in df_exploded.groupby('product')],
                desc="Обработка продуктов"
            ):
                results = process_group(group_data)
                final_result = pd.concat([final_result, pd.DataFrame(results, columns=['product', 'sentence', 'label', 'max_similarity'])], ignore_index=True)

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

        final_result.to_csv(f"./reviews_keywords/automarkup/final_result.csv", mode='a', header=False, index=False)

        # Показать результат
        # display(final_result[['product', 'sentence', 'label', 'max_similarity']])

        # Очистка неиспользуемой памяти перед началом вычислений
        torch.cuda.empty_cache()

        gc.collect()

import cudf.pandas  # Импортирование cuDF и активация его использования
cudf.pandas.install()  # Установка cuDF как основного интерфейса для pandas
import pandas as pd  # Импортирование pandas после установки cuDF
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

final_result = pd.read_csv("./reviews_keywords/automarkup/final_result.csv")
final_result.label.value_counts()
final_result[final_result.label >= 0].label.value_counts()
final_result.loc[final_result.label == 2, "label"] = 0
final_result = final_result[final_result.label >= 0]
display(final_result.describe())
final_result.head()

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

# Очистка неиспользуемой памяти перед началом вычислений
torch.cuda.empty_cache()

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

from transformers import EarlyStoppingCallback

# Измененные параметры для обучения
training_args = TrainingArguments(
    output_dir='./reviews_keywords/results',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./reviews_keywords/logs',
    logging_steps=2,
    evaluation_strategy="steps",
    eval_steps=4,
    fp16=False,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    save_total_limit=2,  # Сохранение только последних двух моделей
    load_best_model_at_end=True,  # Загрузка лучшей модели по окончании обучения
    save_steps=8  # Сохранение модели каждые 1000 шагов
)

# Добавляем EarlyStoppingCallback с терпением в 3 шага
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

# Настройка логирования
logging.basicConfig(filename='./reviews_keywords/clustering.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Создание коллбэка для дополнительного логирования
class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.info(f"Log at step {state.global_step}: {logs}")
            # Дополнительное логирование текущих шагов и состояния
            logging.info(f"Current step: {state.global_step}")
            logging.info(f"Total training steps: {state.max_steps}")
            logging.info(f"Epoch: {state.epoch}")
            logging.info(f"Steps until next evaluation: {state.global_step % args.eval_steps}")
            logging.info(f"Steps until next save: {state.global_step % args.save_steps}")

# Trainer API от Hugging Face
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # Добавление метрик для оценки
    callbacks=[LogCallback(), early_stopping]  # Включение коллбэка для логирования и ранней остановки
)

# Запуск обучения с валидацией
trainer.train()

# Сохранение дообученной модели
model.save_pretrained('./reviews_keywords/fine_tuned_model_10')
tokenizer.save_pretrained('./reviews_keywords/fine_tuned_model_10')

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Получаем предсказания модели на валидационном наборе данных
outputs = trainer.predict(val_dataset)
logits = outputs.predictions
labels = outputs.label_ids

# Функция для вычисления метрик для различных порогов
def evaluate_thresholds(logits, labels, thresholds):
    best_threshold = 0
    best_f1 = 0
    best_metrics = {}
    
    for threshold in thresholds:
        predictions = (logits[:, 1] > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'threshold': threshold
            }
    
    return best_metrics

# Диапазон порогов, который будем проверять
thresholds = np.arange(0.1, 0.9, 0.05)

# Вычисление метрик для различных порогов
best_metrics = evaluate_thresholds(logits, labels, thresholds)

# Вывод лучших метрик и оптимального порога
print(f"Лучший порог: {best_metrics['threshold']}")
print(f"Точность: {best_metrics['accuracy']}")
print(f"F1: {best_metrics['f1']}")
print(f"Precision: {best_metrics['precision']}")
print(f"Recall: {best_metrics['recall']}")
