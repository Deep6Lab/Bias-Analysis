import numpy as np

class WordEmbeddingsAssociationTest:
    def __init__(self, embedding_handler):
        self.embedding_handler = embedding_handler

    def weat_score(self, X, Y, A, B, tokenizer, model, model_name):
        S_X = np.mean([self.embedding_handler.get_embedding(w, tokenizer, model, model_name) for w in X], axis=0)
        S_Y = np.mean([self.embedding_handler.get_embedding(w, tokenizer, model, model_name) for w in Y], axis=0)
        S_A = np.mean([self.embedding_handler.get_embedding(w, tokenizer, model, model_name) for w in A], axis=0)
        S_B = np.mean([self.embedding_handler.get_embedding(w, tokenizer, model, model_name) for w in B], axis=0)
        score = (np.dot(S_X - S_Y, S_A - S_B) / (np.linalg.norm(S_X - S_Y) * np.linalg.norm(S_A - S_B)))
        return score

    def weat_score_with_review(self, X, Y, A, B, tokenizer, model, model_name, review):
        S_X = np.mean([get_embedding(w, tokenizer, model, model_name)[0] for w in X], axis=0)
        S_Y = np.mean([get_embedding(w, tokenizer, model, model_name)[0] for w in Y], axis=0)
        S_A = np.mean([get_embedding(w, tokenizer, model, model_name)[0] for w in A], axis=0)
        S_B = np.mean([get_embedding(w, tokenizer, model, model_name)[0] for w in B], axis=0)
        S_R = np.mean([get_embedding(w, tokenizer, model, model_name)[0] for w in review.split()], axis=0)
        score = (np.dot((S_X - S_Y), (S_A - S_B)) + np.dot(S_R, (S_A - S_B))) / (np.linalg.norm((S_X - S_Y)) * np.linalg.norm((S_A - S_B)))
        return score