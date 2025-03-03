from fastapi import FastAPI
from pydantic import BaseModel
from core.chat import Chatbot

app = FastAPI()
chatbot = Chatbot()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat/")
def chat(request: ChatRequest):
    """Handles chat requests via FastAPI."""
    user_input = request.message
    return {"response": chatbot.process_message(user_input)}

@app.get("/")
def read_root():
    """Root endpoint to check API status."""
    return {"message": "Ask My Docs API is running!"}
