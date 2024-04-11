import random
import pandas as pd
from models import ModelManager
from weat_score import WordEmbeddingsAssociationTest

# word sets of attributes and targets
action_words = ["action", "adventure", "fight", "explosion", "gun"]
romance_words = ["love", "romance", "kiss", "heart", "marriage"]
thriller_words = ["suspense", "mystery", "horror", "thriller", "terror"]
positive_words = ["good", "great", "excellent", "awesome", "fantastic"]
negative_words = ["bad", "terrible", "horrible", "awful", "poor"]

class ScaledBiasAnalysis:
    def __init__(self, data):
        self.data = data
    
    def analyze_scaling_bias(self):
        base_models = ['bert-base-uncased','bert-base-cased','roberta-base','gpt2','gpt2-medium','xlnet-base-cased','albert-base-v2','distilbert-base-uncased','distilbert-base-cased','t5-small','t5-base','google/electra-small-discriminator','google/electra-base-discriminator','microsoft/deberta-base','facebook/bart-base']
        large_models = ['bert-large-uncased','bert-large-cased','roberta-large','gpt2-large','xlnet-large-cased','albert-large-v2','t5-large','google/electra-large-discriminator','microsoft/deberta-large','facebook/bart-large']
        xlarge_models = ['gpt2-xl','albert-xlarge-v2','albert-xxlarge-v2','t5-3b','microsoft/deberta-xlarge']

        categories = {
            'base_models': base_models,
            'large_models': large_models,
            'xlarge_models': xlarge_models
        }

        for category, model_names in categories.items():
            results_df = pd.DataFrame(columns=['Model', 'Action_Score', 'Romance_Score', 'Thriller_Score'])
            for model_name in model_names:
                print(f"Processing {model_name}...")
                model_manager = ModelManager(self.data, model_name)
                tokenizer, model = model_manager.load_model()
                embedding_handler = model_manager
                weat_test = WordEmbeddingsAssociationTest(embedding_handler)
                action_score = weat_test.weat_score(action_words, romance_words + thriller_words, positive_words, negative_words, tokenizer, model, model_name)
                romance_score = weat_test.weat_score(romance_words, action_words + thriller_words, positive_words, negative_words, tokenizer, model, model_name)
                thriller_score = weat_test.weat_score(thriller_words, action_words + romance_words, positive_words, negative_words, tokenizer, model, model_name)

                results_df = results_df.append({
                    'Model': model_name,
                    'Action_Score': action_score,
                    'Romance_Score': romance_score,
                    'Thriller_Score': thriller_score
                }, ignore_index=True)

            results_df.to_csv(f'D:\work-colab\gh-repos\Deep6Lab\Bias-Analysis\results\model_bias_scores_{category}.csv', index=False)
            print(f"Results for {category} saved to model_bias_scores_{category}.csv")
            return

class PromptBiasAnalysis:
    def __init__(self, data):
        self.data = data
    
    def analyze_bias_with_prompts(self):
        df = self.data.copy()
        df['genre'] = df['genre'].str.lower()
        reviews = df.groupby('genre').reset_index(drop=True)

        # prompts
        prompts = {
            "action": "This review is about an action movie. The review says: ",
            "romance": "This review is about a romance movie. The review says: ",
            "thriller": "This review is about a thriller movie. The review says: "
        }

        # list of models for comparison
        model_names = ['bert-base-uncased', 'roberta-base', 'xlnet-base-cased', 't5-small', 'albert-base-v2']

        results = []
        for model_name in model_names:
            model_manager = ModelManager( _ , model_name)
            weat_test = WordEmbeddingsAssociationTest()
            tokenizer, model = model_manager.load_model()
            tokenizer, model = load_model_and_tokenizer(model_name)

            for _, row in reviews.iterrows():
                genre = row['genre']
                review = row['review']
                if genre == 'action':
                    target_words = action_words
                    other_genre_words = romance_words + thriller_words
                elif genre == 'romance':
                    target_words = romance_words
                    other_genre_words = action_words + thriller_words
                elif genre == 'thriller':
                    target_words = thriller_words
                    other_genre_words = action_words + romance_words
                else:
                    continue

                # score without prompt
                score = weat_test.weat_score(target_words, other_genre_words, positive_words, negative_words, tokenizer, model, model_name, review)

                # score with prompt
                prompted_review = prompts.get(genre, "") + review
                prompted_score = weat_test.weat_score_with_review(target_words, other_genre_words, positive_words, negative_words, tokenizer, model, model_name, prompted_review)

                results.append({
                    'Model': model_name,
                    'Genre': genre,
                    'Score': score,
                    'PromptedScore': prompted_score
                })

        return pd.DataFrame(results)