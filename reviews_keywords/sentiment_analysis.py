# sentiment_analysis.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


class SentimentAnalyzer:
    def __init__(self, model_name):
        """
        Инициализация анализатора сентимента.

        :param model_name: Название модели.
        """

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

    def classify(self, text):
        """
        Анализ сентимента текста.

        :param text: Текст для анализа.
        :return: Результат анализа сентимента.
        """
        return self.classifier(text)
