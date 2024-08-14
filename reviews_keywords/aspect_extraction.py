import yake
from .setup_dependencies import setup_dependencies
import yaml
import os
from tqdm import tqdm
import pymorphy2
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from .text_cleaning import clean_text, preprocess_text
from collections import defaultdict, Counter
import re
import logging

# Настройка логирования
logging.basicConfig(
    filename='./reviews_keywords/logfile.log',  # Укажите путь к файлу для сохранения логов
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

CONFIG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yaml')))

nlp = setup_dependencies()
    
# Словари аспектных групп
aspect_groups_all = {
    'основные характеристики': CONFIG['aspect_groups_main'],
    'электроника и устройства': CONFIG['aspect_groups_devices'],
    'автотовары': CONFIG['aspect_groups_cars'],
    'одежда и обувь': CONFIG['aspect_groups_clothes'],
    'бытовая техника': CONFIG['aspect_groups_appliances'],
    'красота и уход': CONFIG['aspect_groups_beauty'],
    'игрушки и детские товары': CONFIG['aspect_groups_toys'],
    'канцелярия и книги': CONFIG['aspect_groups_books'],
    'мебель и интерьер': CONFIG['aspect_groups_furniture'],
    'сад и огород': CONFIG['aspect_groups_garden'],
}

# Инициализация морфологического анализатора для работы с русским языком.
morph = pymorphy2.MorphAnalyzer()

# Настройка KeywordExtractor с параметрами для извлечения ключевых фраз из текста.
kw_extractor = yake.KeywordExtractor(
    lan="ru", 
    n=3,  # Извлечение фраз из трех слов.
    dedupLim=0.9,  # Лимит на дублирование (для фильтрации похожих ключевых слов).
    top=50,  # Количество ключевых слов, которые будут извлекаться.
    dedupFunc="levs"  # Используемая функция для дублирования.
)

def collect_morphology(doc):
    """
    Функция для анализа морфологических характеристик всех слов в документе.
    
    :param doc: Документ для анализа.
    :return: Словарь, где ключ - лемма, значение - наиболее часто встречающиеся морфологические характеристики.
    """
    lemma_stats = defaultdict(lambda: {'gender': Counter(), 'number': Counter(), 'case': Counter(), 
                                       'tense': Counter(), 'person': Counter(), 'mood': Counter(),
                                       'aspect': Counter()})
    
    for token in doc:
        if token.is_stop or token.is_punct:
            continue

        lemma = token.lemma_.lower()

        # Сбор статистики по морфологическим характеристикам
        if 'Gender' in token.morph:
            lemma_stats[lemma]['gender'][token.morph.get('Gender')[0]] += 1
        if 'Number' in token.morph:
            lemma_stats[lemma]['number'][token.morph.get('Number')[0]] += 1
        if 'Case' in token.morph:
            lemma_stats[lemma]['case'][token.morph.get('Case')[0]] += 1
        if 'Tense' in token.morph:
            lemma_stats[lemma]['tense'][token.morph.get('Tense')[0]] += 1
        if 'Person' in token.morph:
            lemma_stats[lemma]['person'][token.morph.get('Person')[0]] += 1
        if 'Mood' in token.morph:
            lemma_stats[lemma]['mood'][token.morph.get('Mood')[0]] += 1
        if 'Aspect' in token.morph:
            lemma_stats[lemma]['aspect'][token.morph.get('Aspect')[0]] += 1

    # Определяем наиболее частые значения для каждой характеристики
    aggregated_stats = {}
    for lemma, stats in lemma_stats.items():
        aggregated_stats[lemma] = {
            'gender': stats['gender'].most_common(1)[0][0] if stats['gender'] else None,
            'number': stats['number'].most_common(1)[0][0] if stats['number'] else None,
            'case': stats['case'].most_common(1)[0][0] if stats['case'] else None,
            'tense': stats['tense'].most_common(1)[0][0] if stats['tense'] else None,
            'person': stats['person'].most_common(1)[0][0] if stats['person'] else None,
            'mood': stats['mood'].most_common(1)[0][0] if stats['mood'] else None,
            'aspect': stats['aspect'].most_common(1)[0][0] if stats['aspect'] else None,
        }

    return aggregated_stats

def map_review_rating(rating):
    """
    Функция для сопоставления рейтинга обзора с категорией настроения.
    
    :param rating: Рейтинг обзора (например, от 1 до 5).
    :return: Возвращает 'positive' для рейтингов 4 и 5, и 'negative' для остальных.
    """
    if rating in [4, 5]:
        return 'positive'
    else:
        return 'negative'

def extract_aspects(text):
    """
    Функция для извлечения ключевых аспектов из текста с использованием YAKE.
    
    :param text: Текст, из которого нужно извлечь аспекты.
    :return: Список ключевых фраз.
    """
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def get_lemmas(doc):
    """
    Функция для получения лемм слов в документе, исключая стоп-слова и знаки препинания.
    
    :param doc: Объект документа Spacy.
    :return: Строка с леммами слов.
    """
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(lemmas)

def collect_related_words(token):
    """
    Рекурсивная функция для сбора связанных слов (дочерние элементы) для данного токена.
    
    :param token: Токен Spacy.
    :return: Список связанных слов.
    """
    related_words = []
    for child in token.children:
        if child.text.lower() == "не":
            for sub_child in child.children:
                if sub_child.pos_ in {"ADJ", "ADV", "VERB"}:
                    related_words.append(f"не_{sub_child.lemma_}")
        elif child.text.lower() in CONFIG["narichiya_stepeni"] or child.text.lower() in CONFIG["sravnitelno_usilitelnye_konstruktsii"]:
            continue
        elif child.pos_ in {"ADJ", "VERB"} and child.lemma_ not in {"хороший", "плохой"}:
            related_words.append(child.lemma_)
        related_words.extend(collect_related_words(child))
    return related_words

def collect_parent_words(token):
    """
    Рекурсивная функция для сбора связанных слов (родительские элементы) для данного токена.
    
    :param token: Токен Spacy.
    :return: Список родительских слов.
    """
    parent_words = []
    if token.head != token:
        if token.head.pos_ in {"ADJ", "VERB"} and token.head.lemma_ not in {"хороший", "плохой"}:
            parent_words.append(token.head.lemma_)
        parent_words.extend(collect_parent_words(token.head))
    return parent_words

def extract_key_thought_sumy(sentences):
    """
    Функция для извлечения ключевой мысли из набора предложений с использованием метода LSA из библиотеки Sumy.
    
    :param sentences: Список предложений.
    :return: Ключевая мысль как строка.
    """
    combined_text = " ".join(sentences)
    parser = PlaintextParser.from_string(combined_text, Tokenizer("russian"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 1)
    return " ".join([str(sentence) for sentence in summary])

# Кэши
group_vectors_cache = {}
category_vectors_cache = {}

def preprocess_category(category):
    """
    Предобрабатывает категорию: удаляет английские слова и разделяет категории по слэшу.

    :param category: Категория продукта.
    :return: Предобработанная категория.
    """
    text = category.replace('/', ' ')
    # Удаляем английские слова
    text = re.sub(r'\b[a-zA-Z]+\b', '', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_group_vector(aspect_group, nlp):
    total_vector = None
    count = 0
    for doc in tqdm(nlp.pipe(aspect_group, disable=["ner", "parser"]), total=len(aspect_group), desc="Вычисление векторов аспектной группы"):
        if total_vector is None:
            total_vector = doc.vector
        else:
            total_vector += doc.vector
        count += 1
    return total_vector / count if count > 0 else None

# Кэш для векторов категорий и групп
category_vectors_cache = {}
group_vectors_cache = {}

import numpy as np

def normalize_vector(vector):
    """
    Нормализует вектор до единичной длины.
    
    :param vector: Вектор для нормализации.
    :return: Нормализованный вектор.
    """
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def get_group_vector(group_name, nlp):
    """
    Возвращает нормализованный вектор для группы на основе её названия.
    
    :param group_name: Название группы аспектов.
    :param nlp: Модель Spacy.
    :return: Нормализованный вектор группы.
    """
    if group_name not in group_vectors_cache:
        doc = nlp(group_name.lower())
        group_vector = doc.vector
        group_vectors_cache[group_name] = normalize_vector(group_vector)
    return group_vectors_cache[group_name]

def get_category_vector(category, nlp):
    """
    Возвращает нормализованный вектор для категории на основе её названия.
    
    :param category: Название категории.
    :param nlp: Модель Spacy.
    :return: Нормализованный вектор категории.
    """
    category_cleaned = preprocess_category(category)
    if category_cleaned not in category_vectors_cache:
        doc = nlp(category_cleaned)
        category_vector = doc.vector
        category_vectors_cache[category_cleaned] = normalize_vector(category_vector)
    return category_vectors_cache[category_cleaned]

def find_best_aspect_groups(category, aspect_groups_all, threshold=0.4):
    best_groups = ['основные характеристики']
    
    category_vector = get_category_vector(category, nlp)
    best_similarity = 0
    best_group_name = None
    
    for group_name in aspect_groups_all.keys():
        group_vector = get_group_vector(group_name, nlp)
        
        similarity = np.dot(category_vector, group_vector)
        logging.debug(f"Similarity between {category} and {group_name}: {similarity}")
        
        if similarity > best_similarity and similarity >= threshold:
            best_similarity = similarity
            best_group_name = group_name
    
    if best_group_name and best_group_name != 'основные характеристики':
        best_groups.append(best_group_name)
    
    logging.info(f"Best groups for category {category}: {best_groups}")
    return best_groups


def find_best_aspect(aspect, best_groups, group_vectors_cache, doc, threshold=0.5):
    aspect_token = [token for token in doc if token.text.lower() == aspect.lower()]
    if not aspect_token:
        return None
    aspect_vector = aspect_token[0].vector
    best_aspect_name = None
    max_similarity = 0
    for group_name in best_groups:
        for aspect_name, keywords in aspect_groups_all[group_name].items():
            group_vector = group_vectors_cache[group_name]
            similarity = aspect_vector @ group_vector
            logging.debug(f"Similarity between aspect {aspect} and {aspect_name}: {similarity}")
            if similarity > max_similarity and similarity >= threshold:
                max_similarity = similarity
                best_aspect_name = aspect_name
    return best_aspect_name if best_aspect_name else aspect

def analyze_aspects(doc, category, rating_sentiment, analyzer):
    text = doc.text
    lemmatized_text = get_lemmas(doc)
    aspects = extract_aspects(lemmatized_text)
    aspect_details = {}
    best_groups = find_best_aspect_groups(category, aspect_groups_all)
    aspect_group_type = None
    for aspect in aspects:
        best_aspect_name = find_best_aspect(aspect, best_groups, group_vectors_cache, doc)
        if best_aspect_name is None:
            continue
        if not aspect_group_type:
            aspect_group_type = ', '.join(best_groups)
        grouped_aspects = aspect_groups_all.get(best_groups[0], {}).get(best_aspect_name, [])
        sentences = [sent for sent in doc.sents if aspect in sent.text]
        if sentences:
            sentiment_result = analyzer.classify(aspect)
            aspect_sentiment = sentiment_result[0]['label']
            combined_sentiment = aspect_sentiment if rating_sentiment == 'neutral' else rating_sentiment
            aspect_desc = max(sentences, key=lambda sent: len([token for token in sent if token.text == aspect])).text
            characteristics = []
            original_combinations = set()
            for token in sentences[0]:
                if token.text == aspect:
                    characteristics.extend(collect_related_words(token))
                    characteristics.extend(collect_parent_words(token))
            characteristics = list(dict.fromkeys(characteristics))
            combined_aspect_characteristics = f"{aspect} {' '.join(characteristics)}"
            description = f"{aspect_desc}. Тональность: {combined_sentiment}. Характеристики: {', '.join(characteristics)}."
            original_combinations.add(aspect_desc)
            aspect_add = None
            if not morph.parse(aspect)[0].tag.POS == "NOUN":
                aspect_add = aspect
            agreed_phrases = []
            for char in characteristics:
                if char not in {"хороший", "плохой"}:
                    agreed_phrases.append(get_agreed_phrase(aspect, char, aspect_add))
            if best_aspect_name not in aspect_details:
                aspect_details[best_aspect_name] = {
                    'sentiment': combined_sentiment,
                    'description': description,
                    'characteristics': characteristics,
                    'original_combinations': original_combinations,
                    'count': 1,
                    'rating_positive': 1 if rating_sentiment == 'positive' else 0,
                    'rating_negative': 1 if rating_sentiment == 'negative' else 0,
                    'agreed_phrases': agreed_phrases,
                    'aspect_add': aspect_add,
                    'aspect_group_type': aspect_group_type,
                    'grouped_aspects': grouped_aspects
                }
            else:
                aspect_details[best_aspect_name]['count'] += 1
                aspect_details[best_aspect_name]['description'] += f" | {description}"
                aspect_details[best_aspect_name]['characteristics'].extend(characteristics)
                aspect_details[best_aspect_name]['original_combinations'].update(original_combinations)
                aspect_details[best_aspect_name]['agreed_phrases'].extend(agreed_phrases)
                if rating_sentiment == 'positive':
                    aspect_details[best_aspect_name]['rating_positive'] += 1
                elif rating_sentiment == 'negative':
                    aspect_details[best_aspect_name]['rating_negative'] += 1
    logging.info(f"Analyzed aspects for category {category}: {aspect_details}")
    return aspect_details

def process_texts(texts, ratings, category, analyzer, batch_size, nlp):
    all_aspect_details = []
    for doc, rating in tqdm(zip(nlp.pipe(texts, batch_size=batch_size), ratings), total=len(texts), desc="Processing docs"):
        rating_sentiment = map_review_rating(rating)
        all_aspect_details.append(analyze_aspects(doc,category, rating_sentiment, analyzer))
    logging.info(f"Processed texts for category {category}: {all_aspect_details}")
    return all_aspect_details

def evaluate_aspect_importance(df):
    """
    Оценивает важность аспектов для каждого продукта на основе данных.

    :param df: DataFrame с данными анализа аспектов.
    :return: Словарь с важностью аспектов для каждого продукта.
    """
    aspect_counter = {}
    for _, row in df.iterrows():
        product = row['product']
        if product not in aspect_counter:
            aspect_counter[product] = {}
        for aspect, details in row['aspect_details'].items():
            if aspect not in aspect_counter[product]:
                aspect_counter[product][aspect] = {
                    'count': 0,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'descriptions': [],
                    'characteristics': [],
                    'original_combinations': set(),
                    'rating_positive': 0,
                    'rating_negative': 0,
                    'sentiment': details['sentiment'],
                    'agreed_phrases': details['agreed_phrases'],
                    'aspect_add': details['aspect_add']
                }
            aspect_counter[product][aspect]['count'] += details['count']
            aspect_counter[product][aspect][details['sentiment']] += 1
            aspect_counter[product][aspect]['descriptions'].append(details['description'])
            aspect_counter[product][aspect]['characteristics'].extend(details['characteristics'])
            aspect_counter[product][aspect]['original_combinations'].update(details['original_combinations'])
            aspect_counter[product][aspect]['rating_positive'] += details['rating_positive']
            aspect_counter[product][aspect]['rating_negative'] += details['rating_negative']
            aspect_counter[product][aspect]['agreed_phrases'].extend(details['agreed_phrases'])
    return aspect_counter

def get_agreed_phrase(*args):
    """
    Согласовывает фразу из набора слов (аспектов и характеристик).

    :param args: Набор слов (аспектов и характеристик).
    :return: Согласованная фраза. Если не получилось согласовать, слова фразы возвращаются через пробел в исходном виде.
    """
    args = [arg for arg in args if arg]  # Удаляем пустые аргументы
    if not args:
        return ""

    try:
        parsed_args = [morph.parse(arg)[0] for arg in args]  # Разбор всех аргументов

        base_word = parsed_args[0]
        agreed_words = [base_word.word]

        for word in parsed_args[1:]:
            if word.tag.POS in {"ADJF", "ADJS", "COMP"} and base_word.tag.POS in {"NOUN", "ADJF", "ADJS", "COMP"}:
                # Согласование прилагательного с существительным или другим прилагательным
                word = word.inflect({base_word.tag.gender, base_word.tag.number, base_word.tag.case})
            elif word.tag.POS == "VERB" and base_word.tag.POS in {"NOUN", "ADJF", "ADJS", "COMP"}:
                # Согласование глагола с существительным или прилагательным
                word = word.inflect({base_word.tag.gender, base_word.tag.number, base_word.tag.case, word.tag.tense, word.tag.person})

            if word:  # Если слово успешно согласовано
                agreed_words.append(word.word)
            else:
                agreed_words.append(word.normal_form)

        return ' '.join(agreed_words)
    except Exception as e:
        return ' '.join(args)  # Если произошла ошибка, возвращаем исходные слова
