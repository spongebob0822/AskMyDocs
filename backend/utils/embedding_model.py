import os
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """Class to handle sentence embeddings using SentenceTransformers."""
    
    def __init__(self, model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, sentences):
        """Generates embeddings for given sentences."""
        return self.model.encode(sentences, convert_to_tensor=False)
