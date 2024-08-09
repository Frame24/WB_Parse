import yake
import yaml
import os
from tqdm import tqdm
import pymorphy2
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from .text_cleaning import clean_text, preprocess_text

CONFIG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yaml')))

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

def analyze_aspects(doc, rating_sentiment, analyzer):
    """
    Анализ аспектов текста, выявление связанных характеристик и определение тональности.

    :param doc: Объект документа Spacy.
    :param rating_sentiment: Сентимент, основанный на рейтинге.
    :param analyzer: Объект для анализа настроения.
    :return: Словарь с детальной информацией об аспектах.
    """
    text = doc.text
    lemmatized_text = get_lemmas(doc)
    aspects = extract_aspects(lemmatized_text)
    aspect_details = {}

    for aspect in aspects:
        aspect_lemma = aspect
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
                aspect = "товар"

            agreed_phrases = []
            for char in characteristics:
                if char not in {"хороший", "плохой"}:
                    if aspect_add:
                        agreed_phrases.append(get_agreed_phrase("товар", char, aspect_add))
                    else:
                        agreed_phrases.append(get_agreed_phrase(aspect, char))

            if aspect_lemma not in aspect_details:
                aspect_details[aspect_lemma] = {
                    'sentiment': combined_sentiment,
                    'description': description,
                    'characteristics': characteristics,
                    'original_combinations': original_combinations,
                    'count': 1,
                    'rating_positive': 1 if rating_sentiment == 'positive' else 0,
                    'rating_negative': 1 if rating_sentiment == 'negative' else 0,
                    'agreed_phrases': agreed_phrases,
                    'aspect_add': aspect_add
                }
            else:
                aspect_details[aspect_lemma]['count'] += 1
                aspect_details[aspect_lemma]['description'] += f" | {description}"
                aspect_details[aspect_lemma]['characteristics'].extend(characteristics)
                aspect_details[aspect_lemma]['original_combinations'].update(original_combinations)
                aspect_details[aspect_lemma]['agreed_phrases'].extend(agreed_phrases)
                if rating_sentiment == 'positive':
                    aspect_details[aspect_lemma]['rating_positive'] += 1
                elif rating_sentiment == 'negative':
                    aspect_details[aspect_lemma]['rating_negative'] += 1

    return aspect_details

def process_texts(texts, ratings, analyzer, batch_size, nlp):
    """
    Обрабатывает набор текстов, извлекает аспекты и определяет их тональность.

    :param texts: Список текстов для анализа.
    :param ratings: Список рейтингов, соответствующих текстам.
    :param analyzer: Объект для анализа настроения.
    :param batch_size: Размер партии для пакетной обработки.
    :param nlp: Объект Spacy для обработки текста.
    :return: Список аспектов с детализированной информацией.
    """
    all_aspect_details = []
    for doc, rating in tqdm(zip(nlp.pipe(texts, batch_size=batch_size), ratings), total=len(texts), desc="Processing docs"):
        rating_sentiment = map_review_rating(rating)
        all_aspect_details.append(analyze_aspects(doc, rating_sentiment, analyzer))
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
    :return: Согласованная фраза. Если не получилось согласовать, слова фразы возвращаются через пробел в исходном виде в формате string 
    """
    args = [arg for arg in args if arg]  # Удаляем пустые аргументы
    try:
        parsed_args = [morph.parse(arg)[0] for arg in args]  # Разбор всех аргументов
        base_word = parsed_args[0]

        for word in parsed_args[1:]:
            if word.tag.POS in {"ADJF", "ADJS", "COMP"} and base_word.tag.POS in {"NOUN", "ADJF", "ADJS", "COMP"}:
                # Если слово прилагательное и базовое слово существительное или прилагательное
                gender = base_word.tag.gender
                number = base_word.tag.number
                gram_case = base_word.tag.case
                word = word.inflect({gender, number, gram_case})

            elif word.tag.POS == "VERB" and base_word.tag.POS in {"NOUN", "ADJF", "ADJS", "COMP"}:
                # Если слово глагол и базовое слово существительное или прилагательное
                gender = base_word.tag.gender
                number = base_word.tag.number
                gram_case = base_word.tag.case
                tense = word.tag.tense
                person = word.tag.person
                word = word.inflect({gender, number, gram_case, tense, person})

            elif word.tag.POS == "ADVB":
                # Если слово наречие
                pass  # Наречие не изменяется

            if word is not None:
                base_word = word

        agreed_phrase = ' '.join(arg if isinstance(arg, str) else arg.word for arg in parsed_args)
        return agreed_phrase
    except Exception as e:
        return ' '.join(args)
