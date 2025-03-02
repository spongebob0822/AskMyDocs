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
