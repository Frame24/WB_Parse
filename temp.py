import os
import pandas as pd
from tqdm import tqdm
import torch
import pyarrow.parquet as pq
import dask.dataframe as dd
import spacy
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertForSequenceClassification, BertConfig
from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import hdbscan
from scipy.spatial.distance import pdist, squareform
import logging
import re
from joblib import Parallel, delayed
from feedbackfuel.models import Review

class ReviewsKeywords:
    def __init__(self, model_path, spacy_model="ru_core_news_lg"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            import cudf.pandas
            cudf.pandas.install()
        self.tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
        self.model = AutoModel.from_pretrained('sberbank-ai/sbert_large_nlu_ru').to(self.device)

        spacy.prefer_gpu()
        self.nlp = spacy.load(spacy_model, disable=["ner", "tagger", "attribute_ruler", "lemmatizer"])
        
        # Загрузка отзывов из базы данных
        self.reviews = Review.objects.all()

    @staticmethod
    def clean_text(text):
        text = re.sub(r'[\n\r\t]+|\s{2,}', ' ', text)
        text = re.sub(r'(?<!\.)\s*\.\s*|\s*\.\s*(?!\.)', '. ', text)
        return text.strip().rstrip('.')

    def split_reviews_into_sentences(self, batch):
        cleaned_texts = [self.clean_text(text) for text in batch['corrected_text']]
        docs = list(self.nlp.pipe(cleaned_texts, batch_size=64))
        batch['sentences'] = [[sent.text for sent in doc.sents] for doc in docs]
        return batch

    def process_reviews(self):
        # Преобразуем отзывы в DataFrame для дальнейшей обработки
        data = {
            'id': [review.id for review in self.reviews],
            'corrected_text': [review.corrected_text for review in self.reviews]
        }
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.split_reviews_into_sentences, batched=True, batch_size=32)
        df = dataset.to_pandas()
        df_exploded = df.explode('sentences').reset_index(drop=True)
        df_exploded = df_exploded.drop(columns=[col for col in df_exploded.columns if col.startswith('__index_level_')])
        return Dataset.from_pandas(df_exploded)


    def compute_sentence_embeddings(self, sentences):
        sentences = [str(sentence) for sentence in sentences if isinstance(sentence, str)]
        if not sentences:
            raise ValueError("Input contains no valid strings.")
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def compute_embeddings_after_explode(self, batch):
        sentences = batch['sentences']
        valid_sentences = [str(sentence) for sentence in sentences if isinstance(sentence, str)]
        if not valid_sentences:
            batch['sentence_embeddings'] = [[]] * len(sentences)
            return batch
        embeddings = self.compute_sentence_embeddings(valid_sentences)
        embeddings = embeddings.astype(np.float32)
        final_embeddings = []
        embed_idx = 0
        for sentence in sentences:
            if isinstance(sentence, str):
                final_embeddings.append(embeddings[embed_idx])
                embed_idx += 1
            else:
                final_embeddings.append(np.zeros(embeddings.shape[1], dtype=np.float32))
        batch['sentence_embeddings'] = final_embeddings
        return batch

    def apply_embeddings(self, dataset_exploded):
        return dataset_exploded.map(self.compute_embeddings_after_explode, batched=True, batch_size=128)

    def extract_key_thought(self, cluster_sentences):
        sentences = cluster_sentences.split(" | ")
        embeddings = self.compute_sentence_embeddings(sentences)
        centroid = np.mean(embeddings, axis=0)
        similarities = cosine_similarity(embeddings, [centroid])
        key_sentence_index = np.argmax(similarities)
        return sentences[key_sentence_index]

    def count_words(self, cluster_sentences):
        words = cluster_sentences.split()
        return len(words)

    def recluster_large_cluster(self, cluster_sentences, eps=0.1, min_samples=2):
        sentences = cluster_sentences.split(" | ")
        embeddings = self.compute_sentence_embeddings(sentences)
        re_clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
        re_cluster_dict = {}
        for idx, label in enumerate(re_clustering.labels_):
            if label == -1:
                continue
            label_str = str(label)
            if label_str not in re_cluster_dict:
                re_cluster_dict[label_str] = []
            re_cluster_dict[label_str].append(sentences[idx])
        return [" | ".join(cluster) for cluster in re_cluster_dict.values()]

    def recursive_clustering(self, cluster_sentences, threshold, eps=0.22, min_samples=3, min_eps=0.02):
        current_eps = eps
        current_min_samples = min_samples
        new_clusters = [cluster_sentences]
        while True:
            next_clusters = []
            reclustered_any = False
            for cluster in new_clusters:
                if self.count_words(cluster) > threshold:
                    while current_eps >= min_eps:
                        reclustered = self.recluster_large_cluster(cluster, eps=current_eps, min_samples=current_min_samples)
                        if len(reclustered) > 1:
                            next_clusters.extend(reclustered)
                            reclustered_any = True
                            break
                        else:
                            if current_eps > min_eps:
                                current_eps -= 0.05
                    if len(reclustered) == 1:
                        next_clusters.append(cluster)
                else:
                    next_clusters.append(cluster)
            new_clusters = next_clusters
            if not reclustered_any:
                break
        return new_clusters

    def generate_predictions(self, dataset_exploded):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Включаем параллелизм токенизатора для ускорения

        tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        model = BertForSequenceClassification.from_pretrained(self.model_path).to(self.device)

        if self.device == torch.device("cuda"):
            model = model.half()

        reviews = dataset_exploded["sentences"]
        reviews = [str(review) for review in reviews if isinstance(review, str) and review.strip()]

        class ReviewDataset(TorchDataset):
            def __init__(self, reviews, tokenizer, max_len=128):
                self.reviews = reviews
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.reviews)

            def __getitem__(self, idx):
                review = self.reviews[idx]
                encoding = self.tokenizer.encode_plus(
                    review,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                return {key: val.flatten() for key, val in encoding.items()}

        dataset = ReviewDataset(reviews, tokenizer)
        batch_size = 32
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        predictions = []

        from torch.cuda.amp import autocast

        for batch in tqdm(dataloader, desc="Предсказание отзывов"):
            batch = {key: val.to(self.device) for key, val in batch.items()}
            
            with torch.no_grad():
                with autocast():  # Используем смешанную точность
                    outputs = model(**batch)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    batch_predictions = (probabilities[:, 1] > 0.7).cpu().numpy()  # Используем порог 0.7
                    predictions.extend(batch_predictions)

        if len(predictions) != len(dataset_exploded):
            print(f"Warning: Length of predictions ({len(predictions)}) does not match length of index ({len(dataset_exploded)})")
            if len(predictions) < len(dataset_exploded):
                missing_count = len(dataset_exploded) - len(predictions)
                predictions.extend([0] * missing_count)
            elif len(predictions) > len(dataset_exploded):
                predictions = predictions[:len(dataset_exploded)]
        dataset_exploded = dataset_exploded.add_column("predictions", predictions)
        return dataset_exploded

    def process_group(self, category_name, product_name, group):
        all_sentences = group['sentences'].tolist()
        if not all_sentences:
            return pd.DataFrame()

        try:
            all_embeddings = self.compute_sentence_embeddings(all_sentences)
        except ValueError as e:
            print(f"Error in computing embeddings for product {product_name}: {e}")
            return pd.DataFrame()

        distance_matrix = squareform(pdist(all_embeddings, metric='cosine'))
        clustering = hdbscan.HDBSCAN(min_samples=3, metric='precomputed').fit(distance_matrix)

        cluster_dict = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue
            label_str = str(label)
            if label_str not in cluster_dict:
                cluster_dict[label_str] = set()
            cluster_dict[label_str].add(all_sentences[idx])

        clusters = [" | ".join(sentences) for sentences in cluster_dict.values()]

        if not clusters:
            return pd.DataFrame()

        group['binary_rating'] = group['review_rating'].apply(lambda x: 1 if x in [4, 5] else 0)
        avg_rating = group['binary_rating'].mean()
        rating_category = 'positive' if avg_rating > 0.7 else 'neutral'
        rating_category = 'neutral' if avg_rating > 0.5 else 'negative'

        threshold = self.determine_threshold(clusters)

        final_clusters = []
        for cluster in clusters:
            if self.count_words(cluster) > threshold:
                final_clusters.extend(self.recursive_clustering(cluster, threshold))
            else:
                final_clusters.append(cluster)

        # Обеспечение минимального количества кластеров
        final_clusters = self.ensure_minimum_clusters(final_clusters, threshold)

        df_exploded_sorted = pd.DataFrame({
            'category': category_name,
            'product': product_name,
            'avg_rating': avg_rating,
            'rating_category': rating_category,
            'cluster_sentences': final_clusters
        })
        df_exploded_sorted['word_count'] = df_exploded_sorted['cluster_sentences'].apply(self.count_words)
        df_exploded_sorted['key_thought'] = df_exploded_sorted['cluster_sentences'].apply(self.extract_key_thought)
        df_exploded_sorted = df_exploded_sorted.sort_values(by='word_count', ascending=False)

        return df_exploded_sorted

    def determine_threshold(self, clusters):
        if len(clusters) == 1:
            cluster_word_count = self.count_words(clusters[0])
            if cluster_word_count > 20:
                return cluster_word_count / 2
            return cluster_word_count
        return np.min([np.mean([self.count_words(cluster) for cluster in clusters]) * 1.5, 250])

    def ensure_minimum_clusters(self, final_clusters, threshold):
        while len(final_clusters) < 3 and any(self.count_words(cluster) > threshold for cluster in final_clusters):
            largest_cluster = max(final_clusters, key=self.count_words)
            final_clusters.remove(largest_cluster)
            new_clusters = self.recursive_clustering(largest_cluster, threshold)
            if len(new_clusters) <= 1:
                final_clusters.append(largest_cluster)
                break
            final_clusters.extend(new_clusters)
        return final_clusters
    
    def cluster_reviews(self, dataset_exploded):
        # Фильтрация на основе предсказаний
        dataset_filtered = dataset_exploded.filter(lambda x: x['predictions'] == 1)
        
        # Преобразование в pandas DataFrame для группировки
        df_filtered = dataset_filtered.to_pandas()
        grouped = df_filtered.groupby(['category', 'product'])

        results = []
        
        # Последовательная обработка без параллелизма
        for (category_name, product_name), group in tqdm(grouped, desc="Processing categories and products"):
            result_df = self.process_group(category_name, product_name, group)
            if not result_df.empty:
                results.append(result_df)

        if results:  # Проверяем, что список results не пуст
            final_result = pd.concat(results, ignore_index=True)
            final_result = final_result[((final_result['word_count'] > 10) & (final_result['key_thought'].str.len() > 5))]
            final_result.to_csv("./reviews_keywords/feedbackfueltest.csv")
        else:
            print("No valid results to concatenate. Returning an empty DataFrame.")
            final_result = pd.DataFrame()  # Возвращаем пустой DataFrame, если нет данных для объединения
        
        return final_result

    def run(self, output_path='./feedbackfuel/results.csv'):
        dataset_exploded = self.process_reviews()
        dataset_exploded = self.apply_embeddings(dataset_exploded)
        dataset_exploded = self.generate_predictions(dataset_exploded)
        result = self.cluster_reviews(dataset_exploded)
        
        # Сохранение результата в CSV-файл
        if not result.empty:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            print("No results to save.")
        
        return result


reviews_keywords = ReviewsKeywords(model_path='./feedbackfuel/fine_tuned_model')
final_result = reviews_keywords.run()