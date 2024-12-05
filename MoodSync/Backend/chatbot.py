import openai
import os

# Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate chatbot responses
def get_chatbot_response(emotion):
    print(f"Received emotion: {emotion}")  # Debug log for emotion received

    # Define prompts based on emotions
    prompts = {
        "happy": "You are feeling happy. Suggest motivational activities to keep the user inspired.",
        "sad": "The user is feeling sad. Offer comforting and uplifting advice.",
        "angry": "The user is feeling angry. Suggest calming and refocusing techniques.",
        "neutral": "The user feels neutral. Provide inspirational and creative suggestions.",
    }

    # Default prompt if the emotion is not recognized
    prompt = prompts.get(emotion, "The user feels undefined. Provide general motivational advice.")

    try:
        # Call OpenAI to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an empathetic and helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        # Extract the chatbot's message
        chatbot_response = response['choices'][0]['message']['content']
        print(f"Chatbot response: {chatbot_response}")  # Debug log for response
        return chatbot_response

    except Exception as e:
        print(f"Error: {e}")  # Debug log for errors
        return "Sorry, I couldn't process your request at the moment."
