import subprocess
import time

# Initialize batch counter
batch_number = 1

while True:
    # Run ImageCapture.py for 1 minute to capture images for the current batch
    print(f"Starting image capture for batch {batch_number}...")
    subprocess.run(["python", "D:\MoodSync\MoodSync\Backend\ImageCapture.py", str(batch_number)])
    print(f"Image capture for batch {batch_number} completed.")

    # Wait for a few seconds to ensure the batch is finalized
    time.sleep(5)

    # Process the captured images using spark.py
    print(f"Processing images for batch {batch_number}...")
    subprocess.run(["python", "D:\MoodSync\MoodSync\Backend\spark.py", str(batch_number)])
    print(f"Processing for batch {batch_number} completed.")

    # Move to the next batch
    batch_number += 1

    # Add a slight delay before starting the next cycle
    print("Waiting to start the next batch...")
    time.sleep(5)


