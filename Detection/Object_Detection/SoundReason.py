import cv2  # For video capture and image processing
import threading  # For parallel execution (e.g., audio recording)
import numpy as np  # Numerical operations
import sounddevice as sd  # Audio recording
import wavio  # Saving audio to WAV files
import librosa  # Audio processing and feature extraction
from tensorflow.keras.models import load_model  # Load pre-trained Keras models
from ultralytics import YOLO  # YOLO for object detection
import os  # For file operations

# تحميل النموذج المدرب لتصنيف الصوت
model_path = 'audio_classifier_mfcc_improved.h5'  # مسار النموذج المدرب
model = load_model(model_path)  # تحميل النموذج المدرب
CATEGORIES = ['Crying', 'Laugh', 'Noise', 'Silence']  # الفئات التي يتم تصنيفها

# مسار نموذج YOLO
model_yolo = YOLO("yolo11n.pt")  # تحميل نموذج YOLO المدرب

# دالة لتسجيل الصوت بشكل غير متزامن
def record_audio_async(duration=6, filename="detected_audio.wav", callback=None):
    """تسجل الصوت لمدة معينة وتحفظه كملف WAV."""
    def _record():
        sample_rate = 44100  # Sampling rate
        channels = 2  # Stereo recording
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')  # Start recording
        sd.wait()  # Wait for recording to finish
        wavio.write(filename, audio, sample_rate, sampwidth=2)  # Save the recorded audio as a WAV file
        print(f"Audio saved to {filename}")  # Debug print to confirm audio is saved
        if callback:  # If a callback function is provided
            callback(filename)  # Process the recorded audio

    threading.Thread(target=_record, daemon=True).start()  # Run the recording function in a separate thread

# دالة لمعالجة الصوت واستخراج MFCC
def process_audio(file_path, duration_seconds=6, target_sr=22050):
    """معالجة الصوت وتحويله إلى MFCC."""
    y, sr = librosa.load(file_path, sr=target_sr)  # Load audio file
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)  # Pad or truncate audio to 6 seconds
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Extract MFCC
    mfcc = np.mean(mfcc.T, axis=0)  # Average over time
    return np.expand_dims(mfcc, axis=-1)  # Add channel dimension

# دالة لتنبؤ سبب بكاء الطفل باستخدام النموذج المدرب
def predict_cry_reason(file_path):
    """تنبؤ سبب بكاء الطفل باستخدام النموذج المدرب."""
    mfcc = process_audio(file_path)  # Process audio into MFCC
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    prediction = model.predict(mfcc)  # Predict using the trained model
    return CATEGORIES[np.argmax(prediction)]  # Return the class with the highest probability

# فئة لإدارة معالجة الصوت في الخلفية
class AudioProcessor:
    def __init__(self):
        self.stop = False  # Flag to stop processing
        self.latest_prediction = "No prediction yet"  # Store the latest prediction
        self.is_processing = False  # Prevent multiple processes from running simultaneously

    def process_audio_file(self, filename):
        """معالجة الملف الصوتي وتحديث التنبؤ الأخير."""
        try:
            print(f"Processing audio file: {filename}")  # Debug print
            prediction = predict_cry_reason(filename)  # Predict cry reason
            print(f"Prediction: {prediction}")  # Debug print
            self.latest_prediction = prediction  # Update the latest prediction
        except Exception as e:
            print(f"Error in audio processing: {e}")  # Print errors if any
        finally:
            if os.path.exists(filename):  # Delete the audio file after processing
                os.remove(filename)
            self.is_processing = False  # Mark processing as complete

    def start_audio_check(self):
        """بدء عملية تسجيل الصوت ومعالجته."""
        if not self.is_processing:  # Only start if no other process is running
            self.is_processing = True
            print("Starting audio check...")  # Debug print
            record_audio_async(duration=6, callback=self.process_audio_file)  # Record and process audio

# فتح الكاميرا
cap = cv2.VideoCapture(0)  # فتح الكاميرا الافتراضية
if not cap.isOpened():
    print("Error: Could not open the camera.")  # رسالة خطأ إذا لم يكن بالإمكان الوصول إلى الكاميرا
    exit()

audio_processor = AudioProcessor()  # تهيئة معالج الصوت

# بدء عملية الفيديو والكشف عن الأجسام
while True:
    ret, frame = cap.read()  # التقاط إطار من الكاميرا
    if not ret:
        print("Error: Could not read a frame from the camera.")  # التعامل مع أخطاء الكاميرا
        break

    # أداء الكشف عن الأجسام باستخدام YOLO
    results = model_yolo(frame)  # الكشف عن الأجسام في الإطار
    detected = False  # علم للكشف عن الهاتف المحمول
    for box in results[0].boxes:  # التكرار عبر الصناديق المكتشفة
        cls = box.cls  # فئة الجسم المكتشف
        conf = box.conf  # درجة الثقة
        if int(cls[0]) == 67 and conf[0] > 0.70:  # إذا تم اكتشاف الهاتف المحمول
            detected = True  # تعيين الكشف إلى صحيح
            break

    # بدء فحص الصوت إذا تم اكتشاف هاتف محمول
    if detected:
        audio_processor.start_audio_check()  # تفعيل تسجيل ومعالجة الصوت

    # عرض التنبؤ الأخير على الفيديو
    cv2.putText(frame, f"Sound Reason: {audio_processor.latest_prediction}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # عرض إطار الفيديو
    cv2.imshow("YOLO Object Detection", frame)

    # الخروج من الفيديو عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تحرير موارد الكاميرا وإغلاق جميع نوافذ OpenCV
cap.release()
cv2.destroyAllWindows()
