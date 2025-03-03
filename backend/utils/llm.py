from dotenv import load_dotenv
import os
import google.generativeai as genai
from collections import deque


class GeminiChatClient:
    """A client to interact with the Gemini AI model with memory management."""

    def __init__(self, model_role: str):
        """Initializes the GeminiChatClient with a custom model role and memory handling."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        genai.configure(api_key=api_key)
        
        self.generation_config = {
            "temperature": 0,
            "top_p": 0,
            "top_k": 40,
            "max_output_tokens": 8000,
            "response_mime_type": "text/plain",
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro", 
            generation_config=self.generation_config
        )
        
        self.chat_history = deque(maxlen=10)  # Store last 10 messages (5 user-bot interactions)
        
        self.system_message = model_role
    
    def send_message(self, user_input: str) -> str:
        """Sends a message to the model, manages memory, and returns the response text."""
        
        # Include system message as context in the first interaction
        if len(self.chat_history) == 0:
            self.chat_history.append({"role": "user", "parts": [self.system_message]})
    
        self.chat_history.append({"role": "user", "parts": [user_input]})
        
        response = self.model.generate_content(list(self.chat_history))
        
        response_text = response.text or "Sorry, I can't answer that."
        self.chat_history.append({"role": "model", "parts": [response_text]})
        
        return response_text



# Example usage
if __name__ == "__main__":
    chatbot = GeminiChatClient("Welcome to Ask My Docs! Type 'exit' to end the chat.")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        response = chatbot.send_message(user_input)
        print(f"Bot: {response}")
