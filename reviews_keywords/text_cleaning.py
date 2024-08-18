# text_cleaning.py
import re
import emoji
from spacy.lang.ru.stop_words import STOP_WORDS as stopwords_ru

def clean_text(text):
    """
    Очищает текст от эмодзи, HTML-тегов и лишних пробелов.

    :param text: Исходный текст.
    :return: Очищенный текст.
    """
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """
    Предобрабатывает текст: удаляет знаки препинания и стоп-слова, приводит к нижнему регистру.

    :param text: Исходный текст.
    :return: Предобработанный текст.
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_ru]
    return ' '.join(tokens)
