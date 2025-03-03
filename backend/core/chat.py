from utils.query_pinecone import QuestionAnswering
from dotenv import load_dotenv
import os
import google.generativeai as genai

class Chatbot:
    """A chatbot that integrates an embedding model and Gemini AI for validation."""
    
    def __init__(self):
        """Initializes the chatbot with the embedding model and Gemini AI."""
        self.qa_system = QuestionAnswering()
        self.gemini_model = self.configure_gemini()
    
    @staticmethod
    def configure_gemini():
        """Configures the Gemini AI model."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 0,
                "top_p": 0,
                "top_k": 40,
                "max_output_tokens": 8000,
                "response_mime_type": "text/plain",
            },
        )
    
    def chat(self):
        """Starts the chatbot interaction loop."""
        print("Welcome to Ask My Docs! Type 'exit' to end the chat.")
        
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                print("Exiting chat. Goodbye!")
                break
            
            answer = self.qa_system.get_answer(user_input)
            validation_response = self.gemini_model.generate_content(
                f"The user asked: '{user_input}'. The given answer is: '{answer}'. "
                "Only answer with 'yes' if the answer correctly responds to the user's question. Otherwise, answer 'no'."
            )

            print(f"Validation Response: {validation_response.text}")
            
            if "yes" in validation_response.text.lower():
                print(f"AskMyDocs: {answer}")
            else:
                print("AskMyDocs: Sorry, I can't answer that.")
    
    def process_message(self, user_input: str) -> str:
        """Processes a single chat message and returns a response."""
        if user_input.lower() == "exit":
            return "Exiting chat. Goodbye!"

        answer = self.qa_system.get_answer(user_input)
        validation_response = self.gemini_model.generate_content(
                f"The user asked: '{user_input}'. The given answer is: '{answer}'. "
                "Only answer with 'yes' if the answer correctly responds to the user's question. Otherwise, answer 'no'."
            )

        if "yes" in validation_response.text.lower():
            return answer
        else:
            return "Sorry, I can't answer that."


# if __name__ == "__main__":
#     chatbot = Chatbot()
#     chatbot.chat()
