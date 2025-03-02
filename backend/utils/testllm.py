from dotenv import load_dotenv
import os
import google.generativeai as genai
from collections import deque


class GeminiChatClient:
    """A client to interact with the Gemini AI model with memory management."""

    def __init__(self):
        """Initializes the GeminiChatClient with configuration settings and memory handling."""
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
        
        self.chat_history = deque(maxlen=10)  # Store last 5 rounds (each round has user and model response)
        
        self.system_message = (
            "You are a helpful and professional customer support assistant for an internet service provider. "
            "If the question or instruction doesn't relate to internet service, respond with: 'Sorry, I can't answer that.'"
        )
    
    def send_message(self, message: str) -> str:
        """Sends a message to the model, manages memory, and returns the response text."""
        if not self._is_relevant(message):
            return "Sorry, I can't answer that."
        
        self.chat_history.append({"role": "user", "parts": [message]})
        
        response = self.model.generate_content(list(self.chat_history))
        
        response_text = response.text or "Sorry, I can't answer that."
        self.chat_history.append({"role": "model", "parts": [response_text]})
        
        return response_text
    
    def _is_relevant(self, message: str) -> bool:
        """Checks if the message is related to internet service."""
        keywords = ["internet", "wifi", "router", "broadband", "ISP", "connection", "speed", "modem"]
        return any(keyword in message.lower() for keyword in keywords)


if __name__ == "__main__":
    client = GeminiChatClient()
    print("Welcome to Gemini Chat! Type 'exit' to end the chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break
        
        response = client.send_message(user_input)
        print(f"Bot: {response}")
