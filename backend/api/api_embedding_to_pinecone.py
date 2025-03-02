import os
from fastapi import FastAPI, UploadFile, File
import shutil
from core.embedding_to_pinecone import PineconeHandler

app = FastAPI()
pinecone_handler = PineconeHandler()

@app.post("/upload-embed/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    
    # Save file temporarily
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process embeddings
    pinecone_handler.embed_and_store(file_path)
    
    return {"message": "File processed and embeddings stored in Pinecone!"}
