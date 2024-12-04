import os
import numpy as np
import pandas as pd
from PIL import Image

def images_to_csv(image_dir, output_csv):
    """
    Converts images in a directory to a CSV file where each row is:
    - Label: The folder name (emotion) as the class.
    - Pixels: Pixel values flattened into a single row.

    Args:
    - image_dir (str): Path to the directory containing image folders for each class.
    - output_csv (str): Path to save the generated CSV file.
    """
    data = []

    # Iterate over each folder (assumes folder names are labels)
    for label in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label)
        if not os.path.isdir(label_path):
            continue  # Skip non-directory files

        print(f"Processing label: {label}")
        # Process each image in the folder
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            # Skip directories and only process image files
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            try:
                # Open and resize image to 48x48 grayscale
                image = Image.open(image_path).convert('L').resize((48, 48))
                pixels = np.array(image).flatten()  # Flatten into a 1D array
                pixel_str = ' '.join(map(str, pixels))  # Convert to space-separated string
                data.append([label, pixel_str])
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['emotion', 'pixels'])
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")

# Main
if __name__ == '__main__':
    # Path to your image directory
    image_directory = '/Users/janyajaiswal/Desktop/ADBMS/MoodSync/MoodSync/data/train'  # Update with your path
    # Path to save the CSV file
    output_csv_path = '/Users/janyajaiswal/Desktop/ADBMS/MoodSync/MoodSync/images.csv'    # Update with your path

    images_to_csv(image_directory, output_csv_path)
