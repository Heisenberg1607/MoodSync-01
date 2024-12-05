import sys
import os
import cv2
import numpy as np
import requests  # Import requests to send data to Flask
from pyspark.sql import SparkSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Get the batch number from the scheduler
batch_number = int(sys.argv[1])

# Initialize SparkSession
spark = SparkSession.builder.appName("ImageProcessing").getOrCreate()

# Load pre-trained facial expression model
model = load_model('D:/MoodSync/MoodSync/Backend/emotion_model_2.h5')  # Replace with your model's path
emotion_labels = ['Happy', 'Neutral', 'Sad', 'happy', 'Neutral']  # Emotion categories

def process_image(image_path):
    """
    Function to process a single image:
    - Detect faces.
    - Predict emotions for each detected face.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    emotions = []
    for (x, y, w, h) in faces:
        # Extract the face and resize it to the model's expected input size
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0  # Normalize pixel values
        face = img_to_array(face)  # Convert to array
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict the emotion
        emotion_probabilities = model.predict(face)[0]
        emotion_label = emotion_labels[np.argmax(emotion_probabilities)]
        emotions.append(emotion_label)

    return emotions

# Locate the folder for the current batch
current_directory = os.path.dirname(os.path.abspath(__file__))
batch_folder = os.path.join(current_directory, f'MinuteBatch_{batch_number}')

# Read image paths from the batch folder
image_paths = [
    os.path.join(batch_folder, f)
    for f in os.listdir(batch_folder)
    if f.endswith('.png')
]

if not image_paths:
    print(f"No images found in {batch_folder}. Exiting.")
    spark.stop()
    exit()

# Parallelize image processing using Spark
rdd = spark.sparkContext.parallelize(image_paths)
results = rdd.map(process_image).collect()

# Aggregate results
emotion_counts = {}
for result in results:
    for emotion in result:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

# Determine the emotion with the highest count
if emotion_counts:
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    print(f"Dominant emotion in batch {batch_number}: {dominant_emotion} ({emotion_counts[dominant_emotion]} occurrences)")

    # Send dominant emotion to app.py
    payload = {
        "batch_number": batch_number,
        "dominant_emotion": dominant_emotion,
        "count": emotion_counts[dominant_emotion]
    }

    # Flask endpoint to send data
    url = "http://127.0.0.1:5000/set-dominant-emotion"

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Successfully sent dominant emotion to server: {response.json()}")
        else:
            print(f"Failed to send dominant emotion. Server responded with: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending dominant emotion to server: {e}")

else:
    print(f"No emotions detected in batch {batch_number}.")

# Stop SparkSession
spark.stop()
