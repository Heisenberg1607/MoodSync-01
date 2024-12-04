import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Load the images.csv file
data_path = 'D:\MoodSync\MoodSync\images.csv'  # Update with your file path
data = pd.read_csv(data_path)

# Map emotion labels to integers
emotion_labels = {label: i for i, label in enumerate(data['emotion'].unique())}
data['emotion'] = data['emotion'].map(emotion_labels)

# Preprocess the pixel data
def preprocess_data(data):
    """
    Converts the 'pixels' column into a numpy array of images and normalizes them.
    """
    images = data['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape(48, 48, 1))
    X = np.stack(images).astype('float32') / 255.0  # Normalize pixel values
    y = data['emotion'].values  # Emotion labels
    return X, y

X, y = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the labels
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=len(emotion_labels))
y_test = to_categorical(y_test, num_classes=len(emotion_labels))

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(emotion_labels), activation='softmax')  # Number of emotion categories
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_emotion_model.keras', monitor='val_loss', save_best_only=True)


# Define a learning rate scheduler
def lr_schedule(epoch):
    initial_lr = 0.001
    decay = 0.1
    return initial_lr * (1 / (1 + decay * epoch))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model using the data generator
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=40,
    callbacks=[early_stopping, checkpoint, lr_scheduler]
)

# Save the final model
model.save('emotion_model_2.h5')
print("Model saved as 'final_emotion_model.h5'.")

# Evaluate on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate and display confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels.keys(), yticklabels=emotion_labels.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
report = classification_report(y_true, y_pred_classes, target_names=emotion_labels.keys())
print("Classification Report:")
print(report)

# Visualize a few predictions
def visualize_predictions(index):
    """
    Visualizes an image with the true and predicted labels.
    """
    image = X_test[index].reshape(48, 48)  # Reshape to 2D
    true_label = list(emotion_labels.keys())[y_true[index]]
    predicted_label = list(emotion_labels.keys())[y_pred_classes[index]]

    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

# Visualize the first 5 test predictions
for i in range(5):
    visualize_predictions(i)

# Plot training results
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Save the training history
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
