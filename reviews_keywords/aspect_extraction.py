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
from collections import defaultdict, Counter

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

def analyze_aspects(doc, rating_sentiment, analyzer):
    text = doc.text
    lemmatized_text = get_lemmas(doc)
    aspects = extract_aspects(lemmatized_text)
    aspect_details = {}

    # Получение морфологических данных для лемм
    morphology_stats = collect_morphology(doc)

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

            agreed_phrases = []
            for char in characteristics:
                if char not in {"хороший", "плохой"}:
                    agreed_phrases.append(get_agreed_phrase(aspect, char, aspect_add))

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
                    'aspect_add': aspect_add,
                    'morphology': morphology_stats.get(aspect_lemma)  # Добавление морфологических данных
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


