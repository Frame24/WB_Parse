import nltk
import spacy
import torch

def setup_dependencies():
    """
    Настраивает зависимости, необходимые для анализа.
    Загружает пакеты NLTK, SpaCy и проверяет доступность CUDA для PyTorch.
    """
    # Загрузка необходимых ресурсов NLTK
    nltk.download('punkt', quiet=True)
    print(f"torch.cuda.is_available() - {torch.cuda.is_available()}")

    # Проверка и загрузка модели spacy
    try:
        nlp = spacy.load("ru_core_news_lg")
    except OSError:
        print("Модель ru_core_news_lg не найдена. Загружаем модель...")
        spacy.cli.download("ru_core_news_lg")
        nlp = spacy.load("ru_core_news_lg")

    return nlp
