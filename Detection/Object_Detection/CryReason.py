import cv2  # For video capture and image processing
import threading  # For parallel execution (e.g., audio recording)
from ultralytics import YOLO  # YOLO for object detection
import sounddevice as sd  # Audio recording
import numpy as np  # Numerical operations
import wavio  # Saving audio to WAV files
import os  # File operations
from keras.models import load_model  # Load pre-trained Keras models
import librosa  # Audio processing and feature extraction

# Load the pre-trained audio classification model
model_path = 'E:/Senior_Project/sound_detection_modelCNN.h5'  # Path to the trained model
model = load_model(model_path)  # Load the trained Keras model
# Define the audio classes for prediction
classes = ['belly_pain', 'burping', 'cold-hot', 'discomfort', "dontKnow", 'hungry', 'lonely', 'scared', 'tired']

# Function to record audio asynchronously
def record_audio_async(duration=6, filename="detected_audio.wav", callback=None):
    """Records audio for a specified duration and saves it as a WAV file."""
    def _record():
        sample_rate = 44100  # Sampling rate
        channels = 2  # Stereo recording
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')  # Start recording
        sd.wait()  # Wait for recording to finish
        wavio.write(filename, audio, sample_rate, sampwidth=2)  # Save the recorded audio as a WAV file
        if callback:  # If a callback function is provided
            callback(filename)  # Process the recorded audio

    threading.Thread(target=_record, daemon=True).start()  # Run the recording function in a separate thread

# Function to process audio and extract mel-spectrogram
def process_audio(file_path, duration_seconds=6, target_sr=22050):
    """Processes an audio file and converts it to mel-spectrogram format."""
    y, sr = librosa.load(file_path, sr=target_sr)  # Load audio file
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)  # Pad or truncate audio to 6 seconds
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # Generate mel-spectrogram
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibel scale
    return mel_spec_db

# Function to predict the reason for the baby's cry
def predict_cry_reason(file_path):
    """Predicts the reason for the baby's cry using the trained model."""
    mel_spec = process_audio(file_path)  # Process audio into mel-spectrogram
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
    mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension
    mel_spec = mel_spec / np.max(mel_spec)  # Normalize data
    prediction = model.predict(mel_spec)  # Predict using the trained model
    return classes[np.argmax(prediction)]  # Return the class with the highest probability

# Class to manage audio processing in the background
class AudioProcessor:
    def __init__(self):
        self.stop = False  # Flag to stop processing
        self.latest_prediction = "No prediction yet"  # Store the latest prediction
        self.is_processing = False  # Prevent multiple processes from running simultaneously

    def process_audio_file(self, filename):
        """Processes an audio file and updates the latest prediction."""
        try:
            prediction = predict_cry_reason(filename)  # Predict cry reason
            self.latest_prediction = prediction  # Update the latest prediction
        except Exception as e:
            print(f"Error in audio processing: {e}")  # Print errors if any
        finally:
            if os.path.exists(filename):  # Delete the audio file after processing
                os.remove(filename)
            self.is_processing = False  # Mark processing as complete

    def start_audio_check(self):
        """Starts the audio recording and processing pipeline."""
        if not self.is_processing:  # Only start if no other process is running
            self.is_processing = True
            record_audio_async(duration=6, callback=self.process_audio_file)  # Record and process audio

# Load the YOLO model for object detection
model_yolo = YOLO("yolo11n.pt")  # Load the YOLO model with a pre-trained configuration

# Open the camera for video capture
cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Error: Could not open the camera.")  # Print error if the camera cannot be accessed
    exit()

audio_processor = AudioProcessor()  # Initialize the audio processor

# Start real-time video and object detection
while True:
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        print("Error: Could not read a frame from the camera.")  # Handle camera errors
        break

    # Perform object detection using YOLO
    results = model_yolo(frame)  # Detect objects in the frame
    detected = False  # Flag for detecting a cell phone
    for box in results[0].boxes:  # Iterate through detected boxes
        cls = box.cls  # Class of the detected object
        conf = box.conf  # Confidence score of the detection
        if int(cls[0]) == 67 and conf[0] > 0.70:  # If detected object is a cell phone with confidence > 70%
            detected = True  # Mark detection as True
            break

    # Start audio checking if a cell phone is detected
    if detected:
        audio_processor.start_audio_check()  # Trigger audio recording and processing

    # Display the latest prediction on the video
    cv2.putText(frame, f"Latest Cry Reason: {audio_processor.latest_prediction}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the video frame
    cv2.imshow("YOLO Object Detection", frame)

    # Quit the video stream when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
