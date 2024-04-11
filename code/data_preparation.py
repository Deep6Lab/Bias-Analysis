import re
import nltk
import pandas as pd
from openai import OpenAI
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

class DataPreprocessor:
    @staticmethod
    def preprocess(text):
        # lowercase
        text = text.lower()

        # remove special characters and digits
        text = re.sub('[^a-zA-Z]+', ' ', text)

        # remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        filtered_tokens = [token for token in tokens if token not in stop_words]
        text = ' '.join(filtered_tokens)

        # remove extra spaces
        text = re.sub('\s+', ' ', text).strip()

        return text

    @staticmethod
    def remove_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        stripped_text = stripped_text.replace('\n', '').replace('\r', '')
        return stripped_text


class DataLabeler:
    def __init__(self, data, openai_api_key, model):
        self.model = model
        self.data = data
        self.api_key = openai_api_key
    
    def gpt3_classify_genre(self, cluster_id, reviews, excluded_genres):
        client = OpenAI(api_key=self.api_key)
        available_genres = [genre for genre in ["action", "romance", "thriller"] if genre not in excluded_genres]

        prompt = f"""
        Given a set of movie reviews, categorize them into one specific genre: {', '.join(available_genres)}.
        Each set of reviews is from a distinct cluster and should correspond to one of these genres.

        Cluster {cluster_id} Reviews:
        {reviews}

        Genre for Cluster {cluster_id} (excluding {', '.join(excluded_genres)}):
        """

        chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt}],model=self.model,)
        if chat_completion.choices:
            response_text = chat_completion.choices[0].message.content
        else:
            response_text = "No response"

        return response_text.strip()

    def label_data(self):
        excluded_genres = []
        df = self.data.copy()
        num_clusters = 3

        for i in range(num_clusters):
            sample_reviews = ' '.join(df[df['cluster'] == i]['review'].sample(50).tolist())
            suggested_genre = gpt3_classify_genre(i, sample_reviews, self.api_key, excluded_genres)
            print(f"Cluster {i}: {suggested_genre}")
            df.loc[df['cluster'] == i, 'genre'] = suggested_genre
            excluded_genres.append(suggested_genre)

        return df


        

        
