from utils.llm import LLM
from utils.query_pinecone import PineconeQuery

class ChatAgent:
    """Handles the main chat interaction and validation."""
    
    def __init__(self):
        self.llm = LLM()
        self.pinecone = PineconeQuery()

    def chat(self):
        """Runs a command-line chat session."""
        print("Hi, how can I help you? (Type 'exit' to quit)")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            context = self.pinecone.query(user_input)
            if not context:
                print("I am sorry, I cannot answer this.")
                continue

            answer = self.llm.generate_answer(user_input, context)

            if self.llm.verify_answer(user_input, answer):
                print(f"AI: {answer}")
            else:
                print("I am sorry, I cannot answer this.")
