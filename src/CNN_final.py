import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# กำหนดโฟลเดอร์ที่มีไฟล์เสียง
directory = 'C:/Users/natty/Desktop/New folder/SoundProject/samplespace'

# เตรียมข้อมูล
mfcc_list = []
labels = []

for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        audio, sr = librosa.load(os.path.join(directory, filename), sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc = np.transpose(mfcc)

        label = filename.split('_')[0]  # สมมุติว่าไฟล์มีรูปแบบ 'dog_xxx.wav'

        for i in range(mfcc.shape[0]):
            mfcc_list.append(mfcc[i])
            labels.append(label)

mfcc_array = np.array(mfcc_list)
labels_array = np.array(labels)

# เข้ารหัส labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_array)

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(mfcc_array, labels_encoded, test_size=0.2, random_state=42)

# ปรับรูปแบบข้อมูลให้เข้ากับ CNN
X_train = X_train.reshape((-1, 13, 1))  # (n_samples, height, width, channels)
X_test = X_test.reshape((-1, 13, 1))

# สร้างโมเดล CNN ด้วย TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 1), activation='relu', input_shape=(13, 1, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    tf.keras.layers.Conv2D(64, (3, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ฝึกโมเดล
num_epochs = 200
model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(X_test, y_test))

# ฟังก์ชันสำหรับการพยากรณ์ตัวอย่างใหม่
def predict_new_sample(file_path):
    if os.path.exists(file_path):
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc = np.transpose(mfcc)

        mfcc_mean = np.mean(mfcc, axis=0).reshape(1, 13, 1)  # ใช้ค่าเฉลี่ยของ MFCC เพื่อให้เป็น input ที่ถูกต้อง
        
        predictions = model.predict(mfcc_mean.reshape(-1, 13, 1, 1))
        predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])[0]
        confidence = np.max(predictions)

        # แสดงค่าความเป็นไปได้สำหรับทุกคลาส
        probabilities = predictions.flatten()

        # กำหนดเกณฑ์ความมั่นใจ
        confidence_threshold = 0.5  # เปลี่ยนตามที่ต้องการ

        if confidence < confidence_threshold:
            print("No matching class found.")  # แสดงข้อความเมื่อไม่เจอคำที่ตรงกัน
            return None, None
        else:
            return predicted_label, confidence
    else:
        print("Error: The specified file does not exist.")
        return None, None

# ตัวอย่างการใช้งาน
inp = "test"
while inp != "exit":
    inp = str(input("Enter name (without file extension, e.g., 'Recording (7)'): "))
    new_sample_path = 'C:/Users/natty/Desktop/New folder/SoundProject/input'
    realpath = os.path.join(new_sample_path, inp + '.wav')  # เพิ่มนามสกุล .wav

    predicted_class, confidence = predict_new_sample(realpath)

    if predicted_class is not None:
        print(f"It looks like: {predicted_class} (Confidence: {confidence * 100:.2f}%)")
    else:
        print("Error: Prediction failed.")