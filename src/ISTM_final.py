import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import noisereduce as nr

# กำหนดโฟลเดอร์ที่มีไฟล์เสียง
directory = 'C:/Users/Chollada/Desktop/New folder/SoundProject/samplespace'

# เตรียมข้อมูล
mfcc_list = []
labels = []
target_length = 100  # ความยาวเป้าหมายสำหรับ MFCC

for filename in os.listdir(directory):
    if filename.endswith('.wav'):
        audio, sr = librosa.load(os.path.join(directory, filename), sr=None)
        audio_denoised = nr.reduce_noise(y=audio, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio_denoised, sr=sr, n_mfcc=13)
        mfcc = np.transpose(mfcc)

        # ถ้า MFCC ยาวเกินไปให้ slice
        if mfcc.shape[0] > target_length:
            mfcc = mfcc[:target_length]
        # ถ้า MFCC สั้นไปให้ pad ด้วย 0
        elif mfcc.shape[0] < target_length:
            padding = np.zeros((target_length - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack((mfcc, padding))

        label = filename.split('_')[0]  # แยก label จากชื่อไฟล์
        mfcc_list.append(mfcc)
        labels.append(label)

# แปลงข้อมูลเป็น array
mfcc_array = np.array(mfcc_list)
labels_array = np.array(labels)

# เข้ารหัส labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_array)

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(mfcc_array, labels_encoded, test_size=0.2, random_state=42)

# ปรับรูปแบบข้อมูลให้เข้ากับ LSTM
X_train = X_train.reshape((-1, target_length, 13))  # (samples, timesteps, features)
X_test = X_test.reshape((-1, target_length, 13))

# สร้างโมเดล LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(target_length, 13), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ฝึกโมเดล
num_epochs = 300
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(X_test, y_test))

# ทำนายบนชุดทดสอบ
y_pred = np.argmax(model.predict(X_test), axis=1)

# คำนวณค่าความแม่นยำ
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy on test data: {accuracy * 100:.2f}%')

# ฟังก์ชันพยากรณ์เสียงใหม่
def predict_new_sample(file_path):
    if os.path.exists(file_path):
        audio, sr = librosa.load(file_path, sr=None)
        audio_denoised = nr.reduce_noise(y=audio, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio_denoised, sr=sr, n_mfcc=13)
        mfcc = np.transpose(mfcc)

        # padding ให้ MFCC มีความยาวเท่ากัน
        if mfcc.shape[0] > target_length:
            mfcc = mfcc[:target_length]
        elif mfcc.shape[0] < target_length:
            padding = np.zeros((target_length - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack((mfcc, padding))

        mfcc = mfcc.reshape(1, target_length, 13)  # ปรับรูปแบบเป็น input ของ LSTM
        
        predictions = model.predict(mfcc)
        predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])[0]
        confidence = np.max(predictions)

        return predicted_label, confidence
    else:
        print("Error: File not found.")
        return None, None

# ตัวอย่างการพยากรณ์
inp = "test"
while inp != "exit":
    inp = str(input("Enter file name (without .wav extension): "))
    new_sample_path = 'C:/Users/Chollada/Desktop/New folder/SoundProject/input/'
    realpath = os.path.join(new_sample_path, inp + '.wav')

    predicted_class, confidence = predict_new_sample(realpath)

    if predicted_class is not None:
        print(f"It looks like: {predicted_class} (Confidence: {confidence * 100:.2f}%)")
    else:
        print("Error: Prediction failed.")
