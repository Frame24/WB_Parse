import os
import yaml
import pandas as pd
from .setup_dependencies import setup_dependencies  # Добавляем этот импорт
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
    :param config_path: Путь к конфигурационному файлу.
    :return: Список кортежей (категория, DataFrame с аспектами, сводка по категории).
    """
        
    nlp = setup_dependencies()
    analyzer = SentimentAnalyzer(CONFIG['model_name'])

    if categories:
        if isinstance(categories, str):
            categories = [categories]
        data = data[data['category'].isin(categories)]
    else:
        categories = data['category'].unique()

    all_aspect_details = []

    for category in categories:
        category_data = data[data['category'] == category]
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
                original_combinations_str = '\n'.join(original_combinations_list)
                if not characteristics_counter:
                    continue
                sorted_characteristics = sorted(characteristics_counter.items(), key=lambda x: x[1], reverse=True)
                sorted_characteristics_str = ', '.join([f"{char}: {count}" for char, count in sorted_characteristics])
                total_characteristics_count = sum(characteristics_counter.values())

                key_thought_sumy = extract_key_thought_sumy(original_combinations_list)

                agreed_phrases = []
                for char, count in sorted_characteristics:
                    if count > 1 and char not in {"хороший", "плохой"}:  # Пропуск "хороший" и "плохой"
                        agreed_phrases.append(get_agreed_phrase('товар', aspect, char))
                        if metrics.get('aspect_add'):
                            agreed_phrases.append(get_agreed_phrase('товар', metrics['aspect_add'], char))

                product_aspects.append({
                    'product': product,
                    'aspect': aspect,
                    'aspect_add': metrics.get('aspect_add'),
                    'count': metrics['count'],
                    'positive': metrics['positive'],
                    'negative': metrics['negative'],
                    'neutral': metrics['neutral'],
                    'descriptions': max(metrics['descriptions'], key=len),
                    'characteristics': sorted_characteristics_str,
                    'original_combinations': original_combinations_str,
                    'total_characteristics_count': total_characteristics_count,
                    'rating_positive': metrics['rating_positive'],
                    'rating_negative': metrics['rating_negative'],
                    'sentiment': 'positive' if metrics['rating_positive'] + metrics['positive'] - metrics['rating_negative'] - metrics['negative'] > 0 else 'negative',
                    'key_thought_sumy': key_thought_sumy,
                    'agreed_phrases': agreed_phrases,
                })

        aspect_df = pd.DataFrame(product_aspects)
        aspect_df = aspect_df.sort_values(by=['product', 'total_characteristics_count'], ascending=[True, False])
        
        # Замена значений в колонке 'aspect' на 'товар', если есть значение в колонке 'aspect_add'
        aspect_df.loc[aspect_df['aspect_add'].notna(), 'aspect'] = 'товар'
        
        category_summary = aspect_df.groupby('product').agg({
            'count': 'sum',
            'positive': 'sum',
            'negative': 'sum',
            'neutral': 'sum',
            'rating_positive': 'sum',
            'rating_negative': 'sum'
        }).reset_index()
        
        def get_top_aspects(df, sentiment, top_n=10):
            filtered_df = df[df['sentiment'] == sentiment]
            agreed_phrases = filtered_df['agreed_phrases'].explode().dropna()
            top_aspects = agreed_phrases.value_counts().head(top_n).index.tolist()
            empty_aspects_str = []
            if not top_aspects:
                empty_aspects = filtered_df[filtered_df['agreed_phrases'].str.len() == 0]['aspect'].tolist()
                if empty_aspects:
                    aspect_str = ', '.join(empty_aspects)
                    if sentiment == 'positive':
                        empty_aspects_str = empty_aspects
                    else:
                        empty_aspects_str = empty_aspects
                else:
                    if sentiment == 'positive':
                        empty_aspects_str = empty_aspects
                    else:
                        empty_aspects_str = empty_aspects
            return ', '.join(top_aspects), empty_aspects_str
        
        category_summary['positive_aspects'], category_summary['positive_empty_aspects'] = zip(*category_summary['product'].apply(lambda x: get_top_aspects(aspect_df[aspect_df['product'] == x], 'positive')))
        category_summary['negative_aspects'], category_summary['negative_empty_aspects'] = zip(*category_summary['product'].apply(lambda x: get_top_aspects(aspect_df[aspect_df['product'] == x], 'negative')))
        
        category_summary['positive_ratio'] = category_summary['positive'] / category_summary['count']
        category_summary['negative_ratio'] = category_summary['negative'] / category_summary['count']
        category_summary = category_summary.sort_values(by='positive_ratio', ascending=False)

        all_aspect_details.append((category, aspect_df, category_summary))
    
    return all_aspect_details
