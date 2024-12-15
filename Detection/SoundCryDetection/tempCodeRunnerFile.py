import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tensorflow.keras.preprocessing import image

# إعداد بيانات التدريب والتحقق
train_data_dir = "E:/Senior_Project/Sound_Detection/spectrograms"
img_height = 224
img_width = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # إعادة قياس القيم إلى النطاق [0, 1]
    validation_split=0.2  # تقسيم البيانات إلى تدريب وتحقيق
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# بناء النموذج
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')  # عدد الفئات بناءً على البيانات
])

model.summary()

# تجميع النموذج
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# تدريب النموذج
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# تقييم النموذج
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# عرض أفضل دقة تحقق
best_val_accuracy = max(history.history['val_accuracy'])
print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")

# رسم منحنيات التدريب والتحقق للخسارة
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# رسم منحنيات التدريب والتحقق للدقة
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# حفظ النموذج المدرب
model.save('baby_Sound_model.h5')
print("Model saved as 'baby_Sound_model.h5'.")

# دالة للتنبؤ بسبب بكاء الطفل باستخدام ملف صوتي
def predict_cry_reason(model, file_path):
    # إعداد الملف الصوتي وتحويله إلى Spectrogram
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.stft(y)
    D = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # تحويل إلى صورة (يمكنك تحسين الأبعاد والخصائص بناءً على احتياجاتك)
    temp_img_path = "temp_spectrogram.png"
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(temp_img_path)
    plt.close()

    # تحميل الصورة كمدخل للنموذج
    img = image.load_img(temp_img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0  # تطبيع الصورة
    img_array = np.expand_dims(img_array, axis=0)

    # توقع السبب
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_labels = list(train_generator.class_indices.keys())
    
    # حذف الصورة المؤقتة بعد استخدامها
    os.remove(temp_img_path)

    return class_labels[predicted_class]

# اختبار النموذج على ملفات جديدة
test_files = [
    'E:/Senior_Project/Sound_Detection/test_dataset/belly-pain/bp3.wav',
    'E:/Senior_Project/Sound_Detection/test_dataset/hungry/hungry1.wav',
    'E:/Senior_Project/Sound_Detection/test_dataset/tired/tired2.wav',
    'E:/Senior_Project/Sound_Detection/test_dataset/discomrt/discomrt3.wav',
    'E:/Senior_Project/Sound_Detection/test_dataset/lonely/lonely2.wav'
]

for file in test_files:
    result = predict_cry_reason(model, file)
    print(f"Predicted Cry Reason for {os.path.basename(file)}: {result}")