import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from fer import FER
from PIL import Image, ImageTk


# Function to start face and emotion detection in video
def start_face_detection_video():
    emotion_detector = FER(mtcnn=True)  # Use MTCNN for better face detection

    # Start capturing video
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces and emotions in the frame
        emotions = emotion_detector.detect_emotions(frame)

        for emotion in emotions:
            # Extract the face coordinates and dominant emotion
            x, y, w, h = emotion['box']
            dominant_emotion = max(emotion['emotions'], key=emotion['emotions'].get)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the detected emotion
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Function to detect emotion in a static image
def detect_emotion_in_image():
    # Open a file dialog to choose the image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;*.png")])

    if file_path:
        # Load the image
        image = cv2.imread(file_path)

        # Initialize the emotion detector
        emotion_detector = FER(mtcnn=True)

        # Detect emotions in the image
        emotions = emotion_detector.detect_emotions(image)

        if emotions:
            for emotion in emotions:
                # Extract the face coordinates and dominant emotion
                x, y, w, h = emotion['box']
                dominant_emotion = max(emotion['emotions'], key=emotion['emotions'].get)

                # Draw rectangle around the face
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Display the detected emotion
                cv2.putText(image, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Show the image with detected emotions
            cv2.imshow('Emotion Detection in Image', image)
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()
        else:
            messagebox.showinfo("Result", "No face detected in the image.")
    else:
        messagebox.showwarning("Input Error", "No image selected!")


# Placeholder for future emotion change functionality
def change_emotion_in_image():
    # Open a file dialog to choose the image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;*.png")])

    if file_path:
        # Load the image
        image = cv2.imread(file_path)

        # Simulate emotion change (this would be where you implement the actual emotion change model like StarGAN)
        messagebox.showinfo("Emotion Change", "Feature to change emotion is coming soon!")

        # Display the original image for now
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        messagebox.showwarning("Input Error", "No image selected!")


# Create a simple Tkinter GUI
root = tk.Tk()
root.title("Emotion Detection GUI")
root.geometry("300x250")

# Label for the GUI
label = tk.Label(root, text="Select an option for emotion detection", font=("Arial", 12))
label.pack(pady=20)

# Button to start video face detection
start_video_button = tk.Button(root, text="Start Video Detection", command=start_face_detection_video, font=("Arial", 12), bg="lightblue")
start_video_button.pack(pady=10)

# Button to detect emotions in an image
image_detection_button = tk.Button(root, text="Detect Emotion in Image", command=detect_emotion_in_image, font=("Arial", 12), bg="lightgreen")
image_detection_button.pack(pady=10)

# Button to change emotion in a static image
change_emotion_button = tk.Button(root, text="Change Emotion in Image", command=change_emotion_in_image, font=("Arial", 12), bg="lightcoral")
change_emotion_button.pack(pady=10)

# Main event loop
root.mainloop()