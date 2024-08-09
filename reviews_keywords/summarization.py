from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def extract_key_thought_sumy(sentences):
    """
    Извлекает ключевые мысли из списка предложений с использованием LSA суммаризатора.

    :param sentences: Список предложений.
    :return: Ключевая мысль в виде строки.
    """
    combined_text = " ".join(sentences)
    parser = PlaintextParser.from_string(combined_text, Tokenizer("russian"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 1)
    return " ".join([str(sentence) for sentence in summary])
