import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt  # Import Matplotlib
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
# Set the paths to the dataset folders
train_dir = r'C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Machine_learning_(Face_emo)\dataset\train'
test_dir = r'C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Machine_learning_(Face_emo)\dataset\test'

# Image preprocessing parameters
img_size = 48
batch_size = 64

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ImageDataGenerator for loading and augmenting the dataset
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Brightness adjustment
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load and preprocess datasets from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define a deeper CNN model
def build_deeper_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))  # Added L2 regularization
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model

# Compile the model
cnn_model = build_deeper_cnn_model()
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Implement learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Get emotion labels from the train generator's class indices
emotion_labels = list(train_generator.class_indices.keys())  # ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Train the model and capture the history
history = cnn_model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=1,  # Adjust the number of epochs as needed
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[reduce_lr]
)

# Model evaluation: Confusion matrix, classification report, precision, recall, and F1-score
y_true = test_generator.classes  # True labels from the test set
y_pred_prob = cnn_model.predict(test_generator)  # Predict probabilities for test set
y_pred = np.argmax(y_pred_prob, axis=1)  # Get the predicted class indices

def plot_confusion_matrix():
    # Get true labels and predicted labels from the test set
    true_labels = test_generator.classes
    predictions = cnn_model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Classification report to show precision, recall, and F1-score for each emotion
class_report = classification_report(y_true, y_pred, target_names=emotion_labels)
print("Classification Report:\n", class_report)

# Optionally, print individual scores for precision, recall, and F1
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Save the model in .keras format
cnn_model.save('emotion-detect.keras')


# Function to plot training history
def plot_training_history():
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Save the model in .keras format
cnn_model.save('emotion_model_from_scratch.keras')

# Load the saved model for emotion detection
emotion_model = tf.keras.models.load_model('emotion_model_from_scratch.keras')

# Emotion labels
emotion_labels = list(train_generator.class_indices.keys())

# Real-time emotion detection in video
def start_face_detection_video():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break if there's an error with the video capture

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            face_roi_resized = cv2.resize(face_roi, (img_size, img_size))
            face_roi_normalized = face_roi_resized / 255.0
            face_roi_reshaped = np.reshape(face_roi_normalized, (1, img_size, img_size, 1))

            emotion_prediction = emotion_model.predict(face_roi_reshaped)
            max_index = np.argmax(emotion_prediction)
            dominant_emotion = emotion_labels[max_index]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Real-time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to detect emotion in an image
def detect_emotion_in_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;*.png")])
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = gray_image[y:y + h, x:x + w]
                face_roi_resized = cv2.resize(face_roi, (img_size, img_size))
                face_roi_normalized = face_roi_resized / 255.0
                face_roi_reshaped = np.reshape(face_roi_normalized, (1, img_size, img_size, 1))

                emotion_prediction = emotion_model.predict(face_roi_reshaped)
                max_index = np.argmax(emotion_prediction)
                dominant_emotion = emotion_labels[max_index]

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('Emotion Detection in Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            messagebox.showerror("Error", "Could not read the image.")

# Placeholder for future emotion change functionality
def change_emotion_in_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;*.png")])
    if file_path:
        image = cv2.imread(file_path)
        messagebox.showinfo("Emotion Change", "Feature to change emotion is coming soon!")
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Create GUI
root = tk.Tk()
root.title("Emotion Detection GUI")
root.geometry("800x600")

# Set background image
background_image = ImageTk.PhotoImage(Image.open("image1.jpg"))
background_label = tk.Label(root, image=background_image)
background_label.place(relx=0, rely=0, relwidth=1, relheight=1)

# Create title label
title_label = tk.Label(root, text="Emotion Detection System", font=("Bodoni MT Black", 24), bg="#f0f0f0")
title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

# Create button frame
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Create buttons
start_video_button = tk.Button(button_frame, text="Start Video Emotion Detection", command=start_face_detection_video, font=("Arial", 14), bg="#4CAF50", fg="white")
start_video_button.pack(pady=10)

detect_image_button = tk.Button(button_frame, text="Detect Emotion in Image", command=detect_emotion_in_image, font=("Arial", 14), bg="#2196F3", fg="white")
detect_image_button.pack(pady=10)

change_emotion_button = tk.Button(button_frame, text="Change Emotion in Image", command=change_emotion_in_image, font=("Arial", 14), bg="#FF9800", fg="white")
change_emotion_button.pack(pady=10)

# Create button for plotting training history
plot_button = tk.Button(button_frame, text="Plot Training History", command=plot_training_history, font=("Arial", 14), bg="#FF5722", fg="white")
plot_button.pack(pady=10)

confusion_matrix_button = tk.Button(button_frame, text="Show Confusion Matrix", command=plot_confusion_matrix, font=("Arial", 14), bg="#9C27B0", fg="white")
confusion_matrix_button.pack(pady=10)
# Create footer label
footer_label = tk.Label(root, text="Developed by:\nAnusha Sundar (620)\nMonish.V (631)\nLakshay Notiyal (661)\nRaghav Pareek (663)", font=("Broadway", 12), bg="#f0f0f0")
footer_label.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

# Start the GUI main loop
root.mainloop()