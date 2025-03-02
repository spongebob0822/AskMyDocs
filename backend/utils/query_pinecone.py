import os
from dotenv import load_dotenv
from utils.embedding_model import EmbeddingModel
from pinecone import Pinecone

class QuestionAnswering:
    """Handles the query processing and retrieving the answer from Pinecone index."""
    
    def __init__(self, model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1", namespace="question_answering"):
        # Load environment variables from .env
        load_dotenv()
        
        # Initialize the embedding model
        self.embedding_model = EmbeddingModel(model_name=model_name)
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Set the namespace
        self.namespace = namespace
        
        # Initialize the Pinecone index
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX"))
    
    def get_answer(self, query):
        """Process the query, retrieve relevant matches and return the answer."""
        # Convert the query into a numerical vector using the embedding model
        query_embedding = self.embedding_model.get_embeddings([query])[0]
        
        # Convert the numpy ndarray to a list for Pinecone
        query_embedding_list = query_embedding.tolist()
        
        # Query Pinecone index
        results = self.index.query(
            namespace=self.namespace,
            vector=query_embedding_list,
            top_k=3,
            include_values=False,
            include_metadata=True
        )
        
        # Extract the most relevant answer from the results
        if results.get("matches"):
            # Assuming the most relevant answer is from the first match
            answer = results["matches"][0]["metadata"].get("answer", "No answer found.")
            return answer
        else:
            return "No matches found."
        
# Example usage
# if __name__ == "__main__":
#     qa_system = QuestionAnswering()
    
#     query = "What color do you like?"
#     answer = qa_system.get_answer(query)
#     print(f"Answer: {answer}")
