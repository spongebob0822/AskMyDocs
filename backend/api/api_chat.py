from fastapi import FastAPI
from core.chat import ChatAgent

app = FastAPI()
chat_agent = ChatAgent()

@app.post("/chat")
async def ask_question(question: str):
    """Handles chat requests via API."""
    context = chat_agent.pinecone.query(question)

    if not context:
        return {"answer": "I am sorry, I cannot answer this."}

    answer = chat_agent.llm.generate_answer(question, context)

    if chat_agent.llm.verify_answer(question, answer):
        return {"question": question, "answer": answer}
    
    return {"answer": "I am sorry, I cannot answer this."}
