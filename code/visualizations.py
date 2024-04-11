import nltk
import random
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

from models import ModelManager


class Phase1Visualization:
    def __init__(self, data):
        self.data = data
    
    def generate_genre_distribution(self):
        genre_totals = self.data['genre'].value_counts().reset_index()
        genre_totals.columns = ['genre', 'counts']
        labels = genre_totals['genre']
        sizes = genre_totals['counts']
        colors = ['gold', 'lightcoral', 'lightskyblue']
        explode = (0.1, 0, 0) 

        fig, ax = plt.subplots(figsize=(8, 4))
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True, pctdistance=0.85)
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        ax.axis('equal')
        plt.title('Overall Genre Distribution', fontsize=14)
        plt.setp(autotexts, size=8, weight="bold", color="black")
        percentages = [f'{s:.1f}%' for s in sizes / sum(sizes) * 100]
        plt.legend(wedges,  [f'{l} - {p}' for l, p in zip(labels, percentages)], title="Genres", loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\genre_distribution.png')
        plt.show()
    
    @staticmethod
    def color_func(genre):
        if genre == "action":
            return f"hsl(20, {random.randint(60, 100)}%, {random.randint(30, 50)}%)"
        elif genre == "romance"
            return f"hsl(330, {random.randint(60, 100)}%, {random.randint(60, 90)}%)"
        else:
            return f"hsl(240, {random.randint(60, 100)}%, {random.randint(30, 50)}%)"

    def generate_word_clouds(self):
        genres = self.data['genre'].unique()
        for genre in genres:
            genre_data = self.data[self.data['genre'] == genre]
            all_reviews = ' '.join(genre_data['review'].tolist())
            words = nltk.word_tokenize(all_reviews)
            word_counts = Counter(word for word in words if word.isalpha() and word not in stop_words) 
            wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=color_func(genre), prefer_horizontal=1.0).generate_from_frequencies(frequencies)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for {genre} Genre')
            plt.savefig(f'D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\word_cloud_{genre}.png', dpi=300)
            plt.show()
    
    @staticmethod
    def vectorize_text(text):
        text = tf.expand_dims(text, -1)
        return tf.squeeze(vectorizer(text))

    def generate_clusters_scatterplot(self):
        max_features = 10000 
        vectorizer = TextVectorization(max_tokens=max_features, output_sequence_length=500)
        vectorizer.adapt(self.data['review'])
        text_vectors = vectorize_text(self.data['review']).numpy()
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(text_vectors)
        tsne = TSNE(n_components=2, random_state=42)
        text_vectors_reduced = tsne.fit_transform(text_vectors)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=text_vectors_reduced[:, 0], y=text_vectors_reduced[:, 1], hue=clusters palette=['gold', 'lightcoral', 'lightskyblue'], alpha=0.6, edgecolor=None)
        plt.title('Clusters of Reviews')
        plt.xlabel('t-SNE Component 1 (Dimensionless)')
        plt.ylabel('t-SNE Component 2 (Dimensionless)')
        plt.legend(title='Cluster')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\clusters_of_reviews_tf.png', dpi=300)
        plt.show()

    def generate_embeddings_clusters(self):
        model_manager = ModelManager(self.data['reviews'], model_name = 'bert-base-uncased')
        bert_tokenizer, bert_model = model_manager.load_model()
        bert_embeddings = model_manager.get_embeddings(bert_tokenizer, bert_model)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(bert_embeddings)
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(bert_embeddings)
        cluster_colors = ['gold', 'lightcoral', 'lightskyblue']
        cluster_labels = ['Cluster 0 - Romance', 'Cluster 1 - Action', 'Cluster 2 - Thriller']
        plt.figure(figsize=(12, 10))
        for i in range(num_clusters=3):
            plt.scatter(reduced_embeddings[clusters == i, 0], reduced_embeddings[clusters == i, 1], c=cluster_colors[i], label=cluster_labels[i], alpha=0.5)
        plt.title('BERT Embeddings Clustered with K-Means', fontsize=18)
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\clusters_of_reviews_bert.png', dpi=300)
        plt.show()


class Phase2Visualization():
    def __init__(self, data, base, large, xlarge):
        self.data = data
        self.base = base
        self.large = large
        self.xlarge = xlarge
    
    @staticmethod
    def get_color(score, scores, hue):
        min_score, max_score = min(scores), max(scores)
        score_norm = (score - min_score) / (max_score - min_score)
        rgb_color = colorsys.hls_to_rgb(hue/360, 0.5 + score_norm * 0.3, 0.5 + score_norm * 0.5)
        return '#%02x%02x%02x' % (int(rgb_color[0]*255), int(rgb_color[1]*255), int(rgb_color[2]*255))

    def score_base_models(self):
        fig, axes = plt.subplots(1, 3, figsize=(21, 8), sharey=True)
        fig.suptitle('Base Model Genre Scores', fontsize=16)
        genres = ['Action', 'Romance', 'Thriller']
        hues = {'Action': 20, 'Romance': 330, 'Thriller': 240}
        for ax, genre in zip(axes, genres):
            genre_scores = self.base[f'{genre}_Score']
            colors = [get_color(score, genre_scores, hues[genre]) for score in genre_scores]
            bars = ax.bar(df_base['Model'], genre_scores, color=colors)
            ax.set_title(f'{genre} Score', fontsize=12)
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.tick_params(axis='x', rotation=90, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',xy=(bar.get_x() + bar.get_width() / 2, height),xytext=(0, 3),textcoords="offset points",ha='center', va='bottom', fontsize=8)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\base_models_score_comparison.png')
        plt.show()

    def score_large_models(self):
        fig, axes = plt.subplots(1, 3, figsize=(21, 8), sharey=True)
        fig.suptitle('Large Model Genre Scores', fontsize=16)
        genres = ['Action', 'Romance', 'Thriller']
        hues = {'Action': 20, 'Romance': 330, 'Thriller': 240}  
        for ax, genre in zip(axes, genres):
            genre_scores = self.large[f'{genre}_Score']
            colors = [get_color(score, genre_scores, hues[genre]) for score in genre_scores]
            bars = ax.bar(df_large['Model'], genre_scores, color=colors)
            ax.set_title(f'{genre} Score', fontsize=12)
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.tick_params(axis='x', rotation=90, labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',xy=(bar.get_x() + bar.get_width() / 2, height),xytext=(0, 3),textcoords="offset points",ha='center', va='bottom', fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\large_models_score_comparison.png')
        plt.show()

    def score_xlarge_models(self):
        fig, axes = plt.subplots(1, 3, figsize=(21, 8), sharey=True)
        fig.suptitle('Extra Large Model Genre Scores', fontsize=16)
        genres = ['Action', 'Romance', 'Thriller']
        hues = {'Action': 20, 'Romance': 330, 'Thriller': 240}
        for ax, genre in zip(axes, genres):
            genre_scores = self.xlarge[f'{genre}_Score']
            colors = [get_color(score, genre_scores, hues[genre]) for score in genre_scores]
            bars = ax.bar(df_xlarge['Model'], genre_scores, color=colors)
            ax.set_title(f'{genre} Score', fontsize=12)
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.tick_params(axis='x', rotation=90, labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',xy=(bar.get_x() + bar.get_width() / 2, height),xytext=(0, 3),textcoords="offset points",ha='center', va='bottom', fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\xlarge_models_score_comparison.png')
        plt.show()

    def score_comparision_categories(self):
        self.data.reset_index(drop=True, inplace=True)
        self.data.drop(columns=['Unnamed: 0'], inplace=True)
        model_category_rotations = {
            'bert': 45,
            'gpt': 0,
            'roberta': 0,
            'albert': 45,
            'deberta': 0,
            'electra': 30,
            'bart': 0,
            'xlnet': 0,
            't5': 0
        }
        colors = {'Action_Score': '#E74C3C', 'Romance_Score': '#FF33E0', 'Thriller_Score': '#3498DB'}
        def adjust_colors(scores, base_color_hex):
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
            base_color = mcolors.hex2color(base_color_hex) + (1,)
            adjusted_colors = [(*base_color[0:3], 0.3 + 0.7 * score) for score in norm_scores]
            return adjusted_colors

        for model_category, rotation in model_category_rotations.items():
            category_df = self.data[self.data['ModelCategory'] == model_category]
            if category_df.empty:
                continue
            fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
            fig.suptitle(f'Score Comparison for {model_category.upper()} Models', fontsize=16)
            for ax, genre in zip(axs, colors.keys()):
                sorted_df = category_df.sort_values(by=genre, ascending=False)
                scores = sorted_df[genre]
                model_names = sorted_df['Model']
                color_intensity = adjust_colors(scores, colors[genre])
                bars = ax.bar(model_names, scores, color=color_intensity)
                ax.set_title(genre.replace('_', ' '))
                ax.set_ylabel("Effect Score")
                ax.tick_params(axis='x', rotation=rotation)
                for bar, score in zip(bars, scores):
                    ax.annotate(f'{score:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            plt.tight_layout()
            plt.savefig(f"D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\score_comparison_{model_category}_models.png")
            plt.show()

