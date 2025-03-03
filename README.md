# AskMyDocs

## Objective
AskMyDocs is a chatbot designed to answer questions based on uploaded files. The current implementation supports CSV files as input.

## Supported File Format
The chatbot can process CSV files with the following format:

```
id,question,answer
1,What is your favourite colour?,blue
```

## Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/spongebob0822/AskMyDocs.git
cd AskMyDocs
```

### 2. Set Up Environment Variables
Create a `.env` file inside the `backend` directory and add the following details:

```
GEMINI_API_KEY = "YOUR GEMINI_API_KEY"
PINECONE_INDEX = 'YOUR PINECONE_INDEX'
PINECONE_API_KEY = "YOUR PINECONE_API_KEY"
```

### 3. Set Up Virtual Environment
```sh
uv venv .venv
. .venv\Scripts\Activate  # On Windows
uv sync
```

## Running the APIs

### Step 1: Start the Embedding API
This API allows users to embed the input file and push it to Pinecone.
```sh
uvicorn api.api_embedding_to_pinecone:app --reload
```

### Step 2: Start the Chat API
This API provides a chatbot interface using FastAPI.
```sh
uvicorn api.api_chat:app --reload
```

## Usage
Once the APIs are running, you can interact with the chatbot by sending POST requests to the `/chat` endpoint with a JSON payload:

### Example Request
```json
{
  "message": "what color she likes?"
}
```

### Example Response
```json
{
  "response": "blue"
}
```

To ask another question, send a new request to the chat API while keeping the server running.

