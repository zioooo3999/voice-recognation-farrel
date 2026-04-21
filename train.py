import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pydub import AudioSegment

# =========================
# 🔄 Convert ke WAV
# =========================
def convert_to_wav(input_file):
    output_file = "temp.wav"
    
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        return output_file
    except Exception as e:
        print("Gagal convert:", input_file, e)
        return None

# =========================
# 🎧 Feature Extraction
# =========================
def extract_features(file):
    try:
        # 🔥 convert dulu ke wav
        wav_file = convert_to_wav(file)
        if wav_file is None:
            return None

        y, sr = librosa.load(wav_file, duration=3)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean

    except Exception as e:
        print(f"Error di file {file}: {e}")
        return None

# =========================
# 📊 Load Dataset
# =========================
X = []
y = []

base_path = os.path.dirname(os.path.abspath(__file__))

for label in ['male', 'female']:
    folder = os.path.join(base_path, "data", label)
    
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        
        features = extract_features(path)
        
        if features is not None:
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Jumlah data:", len(X))

# ❗ Stop kalau data kosong
if len(X) == 0:
    print("Data kosong! Isi dulu folder audio lo.")
    exit()

# =========================
# ✂️ Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 🌲 Train Random Forest
# =========================
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# =========================
# 📈 Evaluasi
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Akurasi:", accuracy)

# =========================
# 🔮 Prediksi file baru
# =========================
test_file = "test.wav"

if os.path.exists(test_file):
    features = extract_features(test_file)
    if features is not None:
        result = model.predict([features])
        print("Hasil prediksi:", result[0])
else:
    print("File test.wav tidak ditemukan")