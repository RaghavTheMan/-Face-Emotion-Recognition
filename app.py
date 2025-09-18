# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import base64
from io import BytesIO

app = Flask(__name__)

# Load pre-trained model
emotion_model = tf.keras.models.load_model('emotion_model_from_scratch.keras')

# Emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_size = 48

# Create uploads directory
if not os.path.exists('uploads'):
    os.makedirs('uploads')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400

    x, y, w, h = faces[0]
    face_roi = gray_image[y:y + h, x:x + w]
    face_roi_resized = cv2.resize(face_roi, (img_size, img_size))
    face_roi_normalized = face_roi_resized / 255.0
    face_roi_reshaped = np.reshape(face_roi_normalized, (1, img_size, img_size, 1))

    emotion_prediction = emotion_model.predict(face_roi_reshaped)
    max_index = np.argmax(emotion_prediction)
    dominant_emotion = emotion_labels[max_index]

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    os.remove(file_path)

    return jsonify({'emotion': dominant_emotion, 'image': image_base64})


@app.route('/process_frame', methods=['POST'])
def process_video_frame():
    data = request.get_json()
    frame_data = data['frame'].split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    detected_faces = []
    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        face_roi_resized = cv2.resize(face_roi, (img_size, img_size))
        face_roi_normalized = face_roi_resized / 255.0
        face_roi_reshaped = np.reshape(face_roi_normalized, (1, img_size, img_size, 1))

        emotion_prediction = emotion_model.predict(face_roi_reshaped)
        dominant_emotion = emotion_labels[np.argmax(emotion_prediction)]

        detected_faces.append({
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'emotion': dominant_emotion
        })

    return jsonify({'faces': detected_faces})


@app.route('/plot_training_history')
def plot_training_history():
    # Load your training history here
    # Example: history = load_training_history()
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot([0.8, 0.85, 0.9], label='Train Accuracy')  # Example data
    plt.plot([0.7, 0.75, 0.8], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([0.4, 0.3, 0.2], label='Train Loss')
    plt.plot([0.5, 0.4, 0.3], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return jsonify({'plot': plot_base64})


@app.route('/plot_confusion_matrix')
def plot_confusion_matrix():
    # Example confusion matrix data
    cm = np.array([[50, 2, 3, 1, 5, 10, 4],
                   [3, 45, 5, 2, 1, 8, 6],
                   [2, 1, 55, 3, 2, 1, 1],
                   [1, 2, 3, 60, 2, 1, 1],
                   [4, 1, 2, 1, 65, 2, 0],
                   [5, 3, 1, 0, 1, 70, 5],
                   [2, 1, 0, 1, 0, 3, 80]])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return jsonify({'plot': plot_base64})


if __name__ == '__main__':
    app.run(debug=True)