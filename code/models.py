import torch
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ModelManager:
    def __init__(self, data, model_name):
        self.model_name = model_name
        self.data = data

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = T5EncoderModel.from_pretrained(self.model_name) if 't5' in self.model_name else AutoModel.from_pretrained(self.model_name)
        return tokenizer, model.to(device)
    
    def get_embeddings(self, tokenizer, model):
        embeddings = []
        model_config = AutoConfig.from_pretrained(self.model_name)
        max_input_length = model_config.max_position_embeddings if hasattr(model_config, "max_position_embeddings") else 512
        for text in self.data:
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_input_length, padding=True).to(device)
                outputs = model(input_ids=inputs['input_ids']) if 't5' in self.model_name else model(**inputs)
                embedded_text = outputs.last_hidden_state.mean(1).cpu().numpy()
                embeddings.append(embedded_text[0])
        return np.array(embeddings)

    @staticmethod
    def kmeans_clusters(embeddings):
        num_clusters = 3 #as we are dealing with three genres
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(embeddings)
        return clusters
