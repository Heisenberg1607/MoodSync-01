from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import subprocess
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Function to run the scheduler
def run_scheduler(batch_number):
    try:
        # Call the scheduler script with the batch number
        subprocess.run(["python", "D:\MoodSync\MoodSync\Backend\scheduler.py", str(batch_number)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running scheduler: {e}")

# Endpoint to start the session
@app.route('/start-session', methods=['POST'])
def start_session():
    data = request.json  # Get JSON data from the frontend
    batch_number = data.get("batch_number", 1)  # Default batch number is 1

    # Run the scheduler in a separate thread
    threading.Thread(target=run_scheduler, args=(batch_number,)).start()

    return jsonify({"status": f"Session started for batch {batch_number}"}), 200

# Endpoint to fetch the dominant emotion
@app.route('/get-dominant-emotion', methods=['GET'])
def get_dominant_emotion():
    # Get the batch number from the query parameter
    batch_number = request.args.get('batch_number', 1)  # Default batch number is 1
    try:
        # Define the path to the emotion file
        emotion_file = f"D:/MoodSync/dominant_emotion_{batch_number}.json"

        # Check if the emotion file exists
        if not os.path.exists(emotion_file):
            return jsonify({"error": f"No emotion data found for batch {batch_number}"}), 404

        # Open and read the emotion data from the file
        with open(emotion_file, "r") as f:
            emotion_data = json.load(f)

        # Return the emotion data as a JSON response
        return jsonify(emotion_data), 200

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({"error": "An error occurred while fetching emotion data", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
