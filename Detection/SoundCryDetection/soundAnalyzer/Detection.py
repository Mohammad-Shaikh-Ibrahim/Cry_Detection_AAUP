import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# مسار البيانات الصوتية
DATA_PATH = 'E:/Senior_Project/Sound_Detection/soundAnalyzer/data'

# الفئات التي يتم تصنيفها
CATEGORIES = ['Crying', 'Laugh', 'Noise', 'Silence']

# استخراج MFCC من الصوت
def extract_mfcc(file_path, n_mfcc=20, duration=6, target_sr=22050):
    """إستخراج MFCC من الملف الصوتي."""
    y, sr = librosa.load(file_path, sr=target_sr, duration=duration)  # تحميل الصوت
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # استخراج MFCC
    return np.mean(mfcc.T, axis=0)  # المتوسط عبر الزمن

# تجهيز البيانات
def prepare_data(data_path, categories):
    X = []  # المدخلات (MFCC)
    y = []  # الفئات
    for category in categories:
        folder_path = os.path.join(data_path, category)
        label = categories.index(category)  # تحويل الفئة إلى رقم
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.wav', '.ogg')):  # التأكد من صيغة الملف
                file_path = os.path.join(folder_path, file_name)
                mfcc = extract_mfcc(file_path)  # استخراج MFCC
                X.append(mfcc)
                y.append(label)
    return np.array(X), np.array(y)

# تحميل البيانات
X, y = prepare_data(DATA_PATH, CATEGORIES)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحويل البيانات إلى تنسيق مناسب للنموذج
X_train = np.expand_dims(X_train, axis=-1)  # إضافة بُعد القناة
X_test = np.expand_dims(X_test, axis=-1)

# تحويل الفئات إلى تمثيل دالة "One-hot Encoding"
y_train = to_categorical(y_train, num_classes=len(CATEGORIES))
y_test = to_categorical(y_test, num_classes=len(CATEGORIES))

# بناء نموذج LSTM باستخدام TensorFlow
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))  # إضافة Dropout لتقليل overfitting
model.add(LSTM(128, return_sequences=True))  # طبقة LSTM إضافية
model.add(Dropout(0.3))  # Dropout إضافي
model.add(BatchNormalization())  # تحسين التدريب باستخدام Batch Normalization
model.add(LSTM(128))  # طبقة LSTM ثالثة
model.add(Dropout(0.3))  # Dropout إضافي
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout إضافي في الطبقة النهائية
model.add(Dense(len(CATEGORIES), activation='softmax'))  # الطبقة النهائية لتصنيف الفئات

# تجميع النموذج
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# حفظ النموذج المدرب
model.save('audio_classifier_mfcc_improved.h5')

# تقييم النموذج
loss, accuracy = model.evaluate(X_test, y_test)
print(f"The model accuracy: {accuracy * 100:.2f}%")

# رسم دقة التدريب والتحقق
import matplotlib.pyplot as plt

# رسم خسارة التدريب والتحقق
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# رسم دقة التدريب والتحقق
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
