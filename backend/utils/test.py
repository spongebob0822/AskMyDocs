from utils.llm import GeminiChatClient

# Define the role for the AI model
model_role = "You are a helpful AI assistant for answering document-based questions."

# Initialize the chat client
chat_client = GeminiChatClient(model_role=model_role)

# Example user queries
user_query1 = "What is the capital of France?"
response1 = chat_client.send_message(user_query1)
print(f"User: {user_query1}")
print(f"Bot: {response1}")

user_query2 = "what is main languague speak in that country?"
response2 = chat_client.send_message(user_query2)
print(f"User: {user_query2}")
print(f"Bot: {response2}")
