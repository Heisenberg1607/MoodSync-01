import cv2
import os
import time
import sys

# Get the batch number from the scheduler
batch_number = int(sys.argv[1])

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a folder for the current batch
current_directory = os.path.dirname(os.path.abspath(__file__))
batch_folder = os.path.join(current_directory, f'MinuteBatch_{batch_number}')
if not os.path.exists(batch_folder):
    os.makedirs(batch_folder)

print(f"Capturing images for batch {batch_number}...")

# Start the 1-minute timer
start_time = time.time()
image_count = 0

while time.time() - start_time < 60:  # Run for 1 minute
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Save the captured frame as an image
    image_filename = os.path.join(batch_folder, f'capture_{image_count}.png')
    cv2.imwrite(image_filename, frame)
    print(f"Image saved: {image_filename}")

    image_count += 1
    time.sleep(0.5)  # Capture one image every 0.5 seconds

print(f"Finished capturing images for batch {batch_number}.")

# Release the webcam
cap.release()
cv2.destroyAllWindows()
