import cv2
import threading
import numpy as np
import sounddevice as sd
import wavio
import librosa
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import os

# تحميل النموذج المدرب الأول لتصنيف الصوت (تحديد إذا كان هناك بكاء)
model_path_audio = 'audio_classifier_mfcc_improved.h5'
model_audio = load_model(model_path_audio)
CATEGORIES = ['Crying', 'Laugh', 'Noise', 'Silence']

# تحميل النموذج المدرب الثاني لتحديد سبب البكاء
model_path_cry_reason = 'E:/Senior_Project/sound_detection_modelCNN.h5'
model_cry_reason = load_model(model_path_cry_reason)
classes = ['belly_pain', 'burping', 'cold-hot', 'discomfort', "dontKnow", 'hungry', 'lonely', 'scared', 'tired']

# مسار نموذج YOLO
model_yolo = YOLO("yolo11n.pt")

# دالة لتسجيل الصوت بشكل غير متزامن
def record_audio_async(duration=6, filename="detected_audio.wav", callback=None):
    def _record():
        sample_rate = 44100
        channels = 2
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()
        wavio.write(filename, audio, sample_rate, sampwidth=2)
        if callback:
            callback(filename)
    threading.Thread(target=_record, daemon=True).start()

# دالة لمعالجة الصوت واستخراج MFCC
def process_audio_mfcc(file_path, duration_seconds=6, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc, axis=-1)

# دالة لتنبؤ إذا كان هناك بكاء باستخدام النموذج المدرب الأول
def predict_cry(file_path):
    mfcc = process_audio_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model_audio.predict(mfcc)
    return CATEGORIES[np.argmax(prediction)]

# دالة لمعالجة الصوت واستخراج mel-spectrogram
def process_audio_mel(file_path, duration_seconds=6, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# دالة لتنبؤ سبب بكاء الطفل باستخدام النموذج المدرب الثاني
def predict_cry_reason(file_path):
    mel_spec = process_audio_mel(file_path)
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    mel_spec = np.expand_dims(mel_spec, axis=0)
    mel_spec = mel_spec / np.max(mel_spec)
    prediction = model_cry_reason.predict(mel_spec)
    return classes[np.argmax(prediction)]

# فئة لإدارة معالجة الصوت في الخلفية
class AudioProcessor:
    def __init__(self):
        self.latest_prediction = "No prediction yet"
        self.cry_prediction = "No Cry"
        self.cry_reason = "No Reason"
        self.is_processing = False

    def process_audio_file(self, filename):
        try:
            # التنبؤ بالبكاء أولاً
            cry_prediction = predict_cry(filename)
            self.cry_prediction = cry_prediction
            if cry_prediction == 'Crying':
                # إذا تم تحديد البكاء، نحدد السبب باستخدام النموذج الثاني
                cry_reason = predict_cry_reason(filename)
                self.cry_reason = cry_reason
            else:
                self.cry_reason = "No crying detected"
        except Exception as e:
            print(f"Error in audio processing: {e}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)
            self.is_processing = False

    def start_audio_check(self):
        if not self.is_processing:
            self.is_processing = True
            record_audio_async(duration=6, callback=self.process_audio_file)

# فتح الكاميرا
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

audio_processor = AudioProcessor()

# بدء عملية الفيديو والكشف عن الأجسام
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read a frame from the camera.")
        break

    # أداء الكشف عن الأجسام باستخدام YOLO
    results = model_yolo(frame)
    detected = False
    for box in results[0].boxes:
        cls = box.cls
        conf = box.conf
        if int(cls[0]) == 67 and conf[0] > 0.70:  # إذا تم اكتشاف الهاتف المحمول
            detected = True
            break

    # بدء فحص الصوت إذا تم اكتشاف هاتف محمول
    if detected:
        audio_processor.start_audio_check()

    # عرض النتيجة الأولى (هل هناك بكاء؟) والنموذج الثاني (سبب البكاء)
    frame_prediction_cry = audio_processor.cry_prediction
    frame_prediction_reason = audio_processor.cry_reason

    # عرض النتيجتين معًا على الفيديو
    cv2.putText(frame, f"Sound Reason: {frame_prediction_cry}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Cry Reason: {frame_prediction_reason}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # عرض إطار الفيديو
    cv2.imshow("YOLO Object Detection", frame)

    # الخروج من الفيديو عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تحرير موارد الكاميرا وإغلاق جميع نوافذ OpenCV
cap.release()
cv2.destroyAllWindows()
