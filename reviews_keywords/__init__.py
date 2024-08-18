from .setup_dependencies import setup_dependencies
from .text_cleaning import clean_text, preprocess_text
from .sentiment_analysis import SentimentAnalyzer
from .aspect_extraction import process_texts, evaluate_aspect_importance, get_agreed_phrase, extract_key_thought_sumy
from .main import analyze_reviews

__all__ = [
    'setup_dependencies', 
    'clean_text', 
    'preprocess_text', 
    'SentimentAnalyzer', 
    'process_texts', 
    'evaluate_aspect_importance', 
    'get_agreed_phrase', 
    'extract_key_thought_sumy',
    'analyze_reviews'
]
