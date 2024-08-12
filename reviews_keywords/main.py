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
    nlp = setup_dependencies()
    analyzer = SentimentAnalyzer(CONFIG['model_name'])

    if categories:
        if isinstance(categories, str):
            categories = [categories]
        data = data[data['category'].isin(categories)]
    else:
        categories = data['category'].unique()

    all_aspect_df = pd.DataFrame()

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
                    'category': category,
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
        
        aspect_df.loc[aspect_df['aspect_add'].notna(), 'aspect'] = 'товар'
        
        all_aspect_df = pd.concat([all_aspect_df, aspect_df], ignore_index=True)

    return all_aspect_df

def create_category_product_summary(all_aspect_df):
    category_product_summary = all_aspect_df.groupby(['category', 'product']).agg({
        'count': 'sum',
        'positive': 'sum',
        'negative': 'sum',
        'neutral': 'sum',
        'rating_positive': 'sum',
        'rating_negative': 'sum'
    }).reset_index()
    
    category_product_summary['positive_ratio'] = category_product_summary['positive'] / category_product_summary['count']
    category_product_summary['negative_ratio'] = category_product_summary['negative'] / category_product_summary['count']

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
    
    category_product_summary['positive_aspects'], category_product_summary['positive_empty_aspects'] = zip(*category_product_summary['product'].apply(lambda x: get_top_aspects(all_aspect_df[all_aspect_df['product'] == x], 'positive')))
    category_product_summary['negative_aspects'], category_product_summary['negative_empty_aspects'] = zip(*category_product_summary['product'].apply(lambda x: get_top_aspects(all_aspect_df[all_aspect_df['product'] == x], 'negative')))
    
    category_product_summary = category_product_summary.sort_values(by='positive_ratio', ascending=False)

    return category_product_summary

def recommend_products(category_product_summary):
    popular_threshold = max(5, int(len(category_product_summary) * 0.25))
    popular_products = category_product_summary.nlargest(popular_threshold, 'count')
    less_popular_products = category_product_summary[(category_product_summary['count'] >= 10) & (~category_product_summary.index.isin(popular_products.index))]

    popular_recommendation = popular_products.iloc[0]
    less_popular_recommendation = less_popular_products.iloc[0] if not less_popular_products.empty else None

    recommendations = {
        'popular': {
            'product': popular_recommendation['product'],
            'positive_ratio': popular_recommendation['positive_ratio'],
            'rating_positive': popular_recommendation['rating_positive'],
            'rating_negative': popular_recommendation['rating_negative'],
            'positive_aspects': popular_recommendation['positive_aspects'],
            'negative_aspects': popular_recommendation['negative_aspects'],
            'positive_empty_aspects': popular_recommendation['positive_empty_aspects'],
            'negative_empty_aspects': popular_recommendation['negative_empty_aspects']
        },
        'less_popular': {
            'product': less_popular_recommendation['product'] if less_popular_recommendation is not None else "N/A",
            'positive_ratio': less_popular_recommendation['positive_ratio'] if less_popular_recommendation is not None else "N/A",
            'rating_positive': less_popular_recommendation['rating_positive'] if less_popular_recommendation is not None else "N/A",
            'rating_negative': less_popular_recommendation['rating_negative'] if less_popular_recommendation is not None else "N/A",
            'positive_aspects': less_popular_recommendation['positive_aspects'] if less_popular_recommendation is not None else "N/A",
            'negative_aspects': less_popular_recommendation['negative_aspects'] if less_popular_recommendation is not None else "N/A",
            'positive_empty_aspects': less_popular_recommendation['positive_empty_aspects'] if less_popular_recommendation is not None else "N/A",
            'negative_empty_aspects': less_popular_recommendation['negative_empty_aspects'] if less_popular_recommendation is not None else "N/A"
        }
    }

    return recommendations

def recommend_pricing(category_product_summary):
    price_recommendations = []

    for _, row in category_product_summary.iterrows():
        if row['positive_ratio'] >= 0.8 and row['negative_ratio'] < 0.1:
            price_change = 0.05  # Рекомендуем поднять цену на 5%
        elif row['positive_ratio'] < 0.5:
            price_change = -0.05  # Рекомендуем понизить цену на 5%
        else:
            price_change = 0  # Оставить цену без изменений

        price_recommendations.append({
            'category': row['category'],
            'product': row['product'],
            'price_change_recommendation': f"{price_change * 100:.2f}%" if price_change != 0 else "No change",
        })

    return pd.DataFrame(price_recommendations)

def generate_product_card(category_product_summary, all_aspect_df):
    product_cards = []

    for _, row in category_product_summary.iterrows():
        product_df = all_aspect_df[(all_aspect_df['category'] == row['category']) & (all_aspect_df['product'] == row['product'])]

        # Название товара
        title = f"{row['product']} - {' '.join(product_df['aspect'].unique())}"

        # Описание товара
        top_positive_aspects = product_df[product_df['sentiment'] == 'positive']['agreed_phrases'].explode().value_counts().head(5)
        description = f"Этот товар имеет следующие положительные характеристики: {', '.join(top_positive_aspects.index)}."

        # Ключевые характеристики
        key_features = ', '.join(product_df['characteristics'].explode().value_counts().head(5).index)

        # Преимущества товара
        advantages = ', '.join(top_positive_aspects.index)

        product_cards.append({
            'category': row['category'],
            'product': row['product'],
            'title': title,
            'description': description,
            'key_features': key_features,
            'advantages': advantages,
        })

    return pd.DataFrame(product_cards)