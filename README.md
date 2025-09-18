# Face Emotion Recognition 🎭

A **deep learning-based system** that detects and classifies human facial emotions in **real-time**.

The project uses **Convolutional Neural Networks (CNN)** trained on grayscale facial images and provides:

- **Training pipeline** for custom datasets
- **Evaluation metrics** (confusion matrix, precision, recall, F1-score)
- **Real-time webcam emotion detection**
- **Image-based emotion detection**
- **Graphical User Interface (GUI)** built with Tkinter
- Future-ready placeholder for **emotion manipulation in images**

---

## 🚀 Features

### Custom Deep CNN Architecture
- 4 convolutional blocks with **Batch Normalization** and **MaxPooling**  
- **L2 regularization** & **Dropout** for better generalization  
- Final **Softmax layer** for **7 emotion classes**

### Advanced Data Augmentation
- Rotation, zoom, horizontal flip, brightness adjustments, etc.

### Real-Time Detection
- Webcam-based emotion recognition using **OpenCV**

### Interactive GUI
- Live video detection  
- Static image detection  
- Training history visualization  
- Confusion matrix visualization  

### Performance Reporting
- Precision, Recall, F1-Score  
- Detailed classification report

### Optimized Training
- **Learning Rate Scheduler** (`ReduceLROnPlateau`)  
- Fully **saved & reloadable model** in `.keras` format

---

## 🧠 Emotion Classes
By default, the system is trained on **7 emotions**:

```

['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

```

---

## 🗂️ Project Structure
```

Face-Emotion-Recognition/
├── app.py                      # Alternate app for emotion detection
├── emotion.py
├── emotion1.py
├── emotion3.py                 # MAIN training & GUI code
├── index.html                  # Web interface (optional)
├── history.py                  # Script for plotting history
├── image1.jpg                   # GUI background image
├── image2.jpg
├── models/
│   ├── emotion\_model\_from\_scratch.keras
│   ├── emotion-detect.keras
│   └── other\_model\_variants...
├── training\_history.csv
├── confusion\_matrix.json
└── dataset/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
├── angry/
├── happy/
└── ...

```

---

## 📊 Workflow Overview

### 1. **Training Phase**
- Preprocesses and augments data using `ImageDataGenerator`.
- Builds and trains a **custom CNN** on the training dataset.
- Saves the best model as `emotion_model_from_scratch.keras`.
- Records training history (`accuracy` & `loss`) for visualization.

---

### 2. **Evaluation Phase**
- Predicts on the test set to calculate:
  - **Confusion Matrix**
  - **Classification Report**
  - **Precision, Recall, and F1-score**
- Visualizes confusion matrix using **Seaborn**.

---

### 3. **Real-Time Detection**
- Opens webcam feed.
- Detects faces using **Haar Cascade Classifier**.
- Classifies detected face into one of the **7 emotions**.
- Draws bounding boxes & emotion labels live on the video stream.

---

### 4. **GUI Functionality**
Buttons to:
- **Start live video detection**
- **Detect emotion in a static image**
- **Plot training history**
- **Show confusion matrix**
- Placeholder for **"Emotion Change"** (future feature)

---

## 🛠️ Requirements

- Python **3.8+**

### Libraries:
```

tensorflow
numpy
opencv-python
matplotlib
seaborn
pandas
scikit-learn
pillow

````

Install all dependencies:
```bash
pip install -r requirements.txt
````

---

## 📥 Dataset

The project expects a **train/test folder structure**:

```
dataset/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    └── ...
```

Each folder should contain grayscale facial images for that emotion class.

> **Note:** Default image size used = `48x48`

---

## ▶️ Running the Project

### 1. **Training the Model**

Run:

```bash
python emotion3.py
```

This will:

* Train the CNN
* Evaluate it
* Save the model
* Launch the **GUI**

---

### 2. **Real-Time Webcam Emotion Detection**

* Run the program
* In the GUI, click **"Start Video Emotion Detection"**
* Press **Q** to quit the webcam feed

---

### 3. **Detect Emotion in a Static Image**

* Use the GUI button **"Detect Emotion in Image"**
* Select an image file (`.jpg`, `.png`)
* The detected face will be highlighted with the predicted emotion

---

## 🧩 Key Functions in `emotion3.py`

| **Function**                   | **Purpose**                                                             |
| ------------------------------ | ----------------------------------------------------------------------- |
| `build_deeper_cnn_model()`     | Builds the CNN model with 4 convolutional blocks and L2 regularization. |
| `plot_training_history()`      | Visualizes training vs. validation accuracy and loss.                   |
| `plot_confusion_matrix()`      | Displays confusion matrix for test set predictions.                     |
| `start_face_detection_video()` | Real-time webcam emotion detection.                                     |
| `detect_emotion_in_image()`    | Detects emotions in a single uploaded image.                            |
| `change_emotion_in_image()`    | Placeholder for future emotion editing feature.                         |

---

## 📈 Evaluation Metrics

After training, the model prints:

* **Classification Report** with precision, recall, and F1-score per class.
* Weighted averages for overall performance.
* Confusion Matrix Heatmap.

Example output:

```
Precision: 0.8921
Recall:    0.8845
F1 Score:  0.8873
```

---

## 📊 Sample Plots

### **Training Accuracy & Loss**

Shows how the model performed across epochs.

### **Confusion Matrix**

Visualizes per-class performance:

> **True Labels vs Predicted Labels**

---

## 🌟 Future Improvements

* Integrate **transfer learning** with pre-trained models like ResNet or MobileNet.
* Add **real-time emotion trend tracking** over video streams.
* Implement the **"Emotion Change"** feature for image manipulation.
* Optimize model for **mobile deployment** using TensorFlow Lite.

---

## ✍️ Authors

Developed by:

* **Anusha Sundar (620)**
* **Monish V (631)**
* **Lakshay Notiyal (661)**
* **Raghav Pareek (663)**

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 💡 Tips

* Adjust **epochs** in the `cnn_model.fit()` call inside `emotion3.py` for longer training.
* Modify `img_size` or dataset augmentation settings to experiment with performance.

---

## 📌 Main Script Reminder

> **`emotion3.py`** is the **core script** for training, evaluation, real-time detection, and launching the GUI.

---

## Example Run

```bash
python emotion3.py
```
