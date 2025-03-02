import os
import google.generativeai as genai
from dotenv import load_dotenv

class LLM:
    """Class to interact with Gemini (Google's Generative AI)."""

    def __init__(self, api_key=None, model_name="learnlm-1.5-pro-experimental"):
        # Configure the Gemini API with the provided key
        load_dotenv()
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
        
        genai.configure(api_key=api_key)

        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
        )

        self.chat_session = self.model.start_chat(
            history=[]
        )

    def ask(self, input_text):
        """Send a message to the Gemini model and get the response."""
        # Send the user input as a message
        response = self.chat_session.send_message(input_text)
        return response.text
