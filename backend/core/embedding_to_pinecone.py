import os
from dotenv import load_dotenv
import csv
import json
from pinecone import Pinecone
from utils.embedding_model import EmbeddingModel

class PineconeHandler:
    """Handles Pinecone operations including inserting embeddings."""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("PINECONE_API_KEY", "your-api-key")
        self.index_name = os.getenv("PINECONE_INDEX", "my-embeddings-index")
        
        self.pc = Pinecone(api_key=self.api_key)

        # Ensure the index exists
        if self.index_name in self.pc.list_indexes().names():
            self.index = self.pc.Index(self.index_name)
        else:
            raise ValueError(f"Index '{self.index_name}' not found. Please create it first.")

        self.embedding_model = EmbeddingModel()

    def embed_and_store(self, file_path):
        """Reads a file, generates embeddings, and stores them in Pinecone."""
        data = []
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)

        # Prepare text for embedding
        sentences = [f"{row['question']} {row['answer']}" for row in data]
        embeddings = self.embedding_model.get_embeddings(sentences)

        # Insert into Pinecone
        vectors = [
            (row["id"], embedding.tolist(), {"question": row["question"], "answer": row["answer"]})
            for row, embedding in zip(data, embeddings)
        ]
        self.index.upsert(vectors, namespace="question_answering")

        print(f"Inserted {len(vectors)} embeddings into Pinecone.")

# if __name__ == "__main__":
#     handler = PineconeHandler()
    
#     # Get the base directory (backend/core)
#     base_dir = os.path.dirname(os.path.abspath(__file__))  
    
#     # Navigate to the `data/input_file` folder
#     file_path = os.path.join(base_dir, "..", "..", "data", "input_file", "input.csv")
    
#     # Normalize the path for cross-platform compatibility
#     file_path = os.path.normpath(file_path)  

#     handler.embed_and_store(file_path)

