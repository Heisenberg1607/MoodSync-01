from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import get_chatbot_response

import threading
import subprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# In-memory storage for emotions
dominant_emotions = {}

# Function to run the scheduler
def run_scheduler(batch_number):
    try:
        # Call the scheduler script with the batch number
        subprocess.run(["python", "D:/MoodSync/MoodSync/Backend/scheduler.py", str(batch_number)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running scheduler for batch {batch_number}: {e}")

# Endpoint to start the session
@app.route('/start-session', methods=['POST'])
def start_session():
    data = request.json
    batch_number = data.get("batch_number", 1)

    # Start the scheduler in a separate thread
    threading.Thread(target=run_scheduler, args=(batch_number,)).start()

    return jsonify({"status": f"Session started for batch {batch_number}"}), 200

# Endpoint to set the dominant emotion (called by spark.py)
@app.route('/set-dominant-emotion', methods=['POST'])
def set_dominant_emotion():
    data = request.json
    print(f"Received from spark.py: {data}")  # Debugging log for incoming data
    batch_number = data.get("batch_number")
    dominant_emotion = data.get("dominant_emotion")
    count = data.get("count")

    if not batch_number or not dominant_emotion:
        return jsonify({"error": "Invalid data received"}), 400

    # Store the data in memory
    dominant_emotions[batch_number] = {
        "batch_number": batch_number,
        "dominant_emotion": dominant_emotion,
        "count": count
    }
    print(f"Stored dominant emotion for batch {batch_number}: {dominant_emotions[batch_number]}")
    return jsonify({"status": "Dominant emotion stored successfully"}), 200

# Endpoint to fetch the dominant emotion
@app.route('/get-dominant-emotion', methods=['GET'])
def get_dominant_emotion():
    batch_number = int(request.args.get('batch_number', 1))

    if batch_number not in dominant_emotions:
        return jsonify({"error": f"No emotion data found for batch {batch_number}"}), 404

    print(f"Fetched dominant emotion for batch {batch_number}: {dominant_emotions[batch_number]}")  # Debug log
    return jsonify(dominant_emotions[batch_number]), 200

# Chatbot endpoint
@app.route('/get-chat-response', methods=['POST'])
def get_chat_response():
    data = request.json
    print(f"Chatbot triggered with emotion: {data}")  # Debug log
    emotion = data.get("emotion")

    if not emotion:
        return jsonify({"error": "Emotion not provided"}), 400

    # Call the chatbot function
    chatbot_response = get_chatbot_response(emotion)
    print(f"Chatbot response: {chatbot_response}")  # Debug log

    # Return the chatbot's response
    return jsonify({"message": chatbot_response}), 200

if __name__ == '__main__':
    app.run(debug=True)
