import os
import yaml
import pandas as pd
from .setup_dependencies import setup_dependencies
from .text_cleaning import clean_text, preprocess_text
from .sentiment_analysis import SentimentAnalyzer
from .aspect_extraction import map_review_rating, process_texts, evaluate_aspect_importance, get_agreed_phrase, extract_key_thought_sumy
from nltk.tokenize import sent_tokenize

CONFIG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yaml')))

def analyze_reviews(data, categories=None):
    """
    Анализирует отзывы и извлекает важные аспекты для указанных категорий.

    :param data: DataFrame с данными отзывов.
    :param categories: Список категорий для анализа (или None для анализа всех категорий).
    :return: DataFrame с объединенными результатами анализа по всем категориям.
    """
    
    nlp = setup_dependencies()
    analyzer = SentimentAnalyzer(CONFIG['model_name'])

    if categories:
        if isinstance(categories, str):
            categories = [categories]
        data = data[data['category'].isin(categories)]
    else:
        categories = data['category'].unique()

    all_aspect_df = pd.DataFrame()  # Инициализация пустого DataFrame для всех результатов

    for category in categories:
        category_data = data.loc[data['category'] == category].copy()
        category_data['processed_text'] = category_data['corrected_text'].apply(preprocess_text)
        category_data['sentences'] = category_data['corrected_text'].apply(sent_tokenize)
        category_data['rating_sentiment'] = category_data['review_rating'].apply(map_review_rating)

        texts = category_data["corrected_text"].apply(clean_text).tolist()
        ratings = category_data["review_rating"].tolist()

        aspect_details = process_texts(texts, ratings, analyzer, CONFIG['batch_size'], nlp)
        category_data['aspect_details'] = aspect_details

        aspect_importance = evaluate_aspect_importance(category_data)

        product_aspects = []
        for product, aspects in aspect_importance.items():
            for aspect, metrics in aspects.items():
                if metrics['count'] < 3:
                    continue  
                characteristics_counter = {char: metrics['characteristics'].count(char) for char in set(metrics['characteristics'])}
                original_combinations_list = list(set(metrics['original_combinations']))
                if not characteristics_counter:
                    continue
                sorted_characteristics = sorted(characteristics_counter.items(), key=lambda x: x[1], reverse=True)
                sorted_characteristics_str = ', '.join([f"{char}: {count}" for char, count in sorted_characteristics])
                total_characteristics_count = sum(characteristics_counter.values())

                key_thought_sumy = extract_key_thought_sumy(original_combinations_list)

                product_aspects.append({
                    'category': category,  # Добавляем категорию в итоговый DataFrame
                    'product': product,
                    'aspect': aspect,
                    'aspect_add': metrics.get('aspect_add'),
                    'count': metrics['count'],
                    'positive': metrics['positive'],
                    'negative': metrics['negative'],
                    'neutral': metrics['neutral'],
                    'descriptions': max(metrics['descriptions'], key=len),
                    'characteristics': sorted_characteristics_str,
                    'original_combinations': original_combinations_list,
                    'total_characteristics_count': total_characteristics_count,
                    'rating_positive': metrics['rating_positive'],
                    'rating_negative': metrics['rating_negative'],
                    'sentiment': 'positive' if metrics['rating_positive'] + metrics['positive'] - metrics['rating_negative'] - metrics['negative'] > 0 else 'negative',
                    'key_thought_sumy': key_thought_sumy,
                    'agreed_phrases': metrics["agreed_phrases"],
                })

        aspect_df = pd.DataFrame(product_aspects)
        aspect_df = aspect_df.sort_values(by=['product', 'total_characteristics_count'], ascending=[True, False])
        
        # Замена значений в колонке 'aspect' на 'товар', если есть значение в колонке 'aspect_add'
        aspect_df.loc[aspect_df['aspect_add'].notna(), 'aspect'] = 'товар'
        
        # Объединяем текущий DataFrame с общим
        all_aspect_df = pd.concat([all_aspect_df, aspect_df], ignore_index=True)

    return all_aspect_df

def create_category_summary(all_aspect_df):
    """
    Создает сводку по категориям на основе объединенного DataFrame с аспектами.

    :param all_aspect_df: DataFrame, содержащий данные по всем категориям.
    :return: DataFrame со сводной информацией по категориям.
    """
    # Группировка по категориям и расчет суммарных показателей
    category_summary = all_aspect_df.groupby('category').agg({
        'count': 'sum',
        'positive': 'sum',
        'negative': 'sum',
        'neutral': 'sum',
        'rating_positive': 'sum',
        'rating_negative': 'sum'
    }).reset_index()
    
    # Подсчет отношения положительных и отрицательных отзывов
    category_summary['positive_ratio'] = category_summary['positive'] / category_summary['count']
    category_summary['negative_ratio'] = category_summary['negative'] / category_summary['count']

    def get_top_aspects(df, sentiment, top_n=10):
        filtered_df = df[df['sentiment'] == sentiment]
        agreed_phrases = filtered_df['agreed_phrases'].explode().dropna()
        top_aspects = agreed_phrases.value_counts().head(top_n).index.tolist()
        empty_aspects_str = []
        if not top_aspects:
            empty_aspects = filtered_df[filtered_df['agreed_phrases'].str.len() == 0]['aspect'].tolist()
            if empty_aspects:
                aspect_str = ', '.join(empty_aspects)
                empty_aspects_str = empty_aspects
        return ', '.join(top_aspects), empty_aspects_str
    
    # Получение топ аспектов и пустых аспектов для каждой категории
    category_summary['positive_aspects'], category_summary['positive_empty_aspects'] = zip(*category_summary['category'].apply(lambda x: get_top_aspects(all_aspect_df[all_aspect_df['category'] == x], 'positive')))
    category_summary['negative_aspects'], category_summary['negative_empty_aspects'] = zip(*category_summary['category'].apply(lambda x: get_top_aspects(all_aspect_df[all_aspect_df['category'] == x], 'negative')))
    
    # Сортировка по соотношению положительных отзывов
    category_summary = category_summary.sort_values(by 'positive_ratio', ascending=False)

    return category_summary
