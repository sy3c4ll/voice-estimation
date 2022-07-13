import librosa
import numpy as np
import pyaudio
import sys
from PyQt6.QtWidgets import QApplication, QPushButton
from tensorflow.keras.models import load_model

model_path = "models/"
sample_rate = 48000
duration = 3
chunk = 1200

def predict():
  global app, button

  # Start recording
  print("[*] Recording started")
  stream = pa.open(format = pyaudio.paFloat32, channels = 1, rate = sample_rate, frames_per_buffer = chunk, input = True, output = False)
  audio_data = np.array([])
  for i in range(int(sample_rate * duration / chunk)):
    if i % int(sample_rate / chunk) == 0:
      button.setText(str(3 - i // int(sample_rate / chunk)))
      app.processEvents()
    audio_data = np.append(audio_data, np.frombuffer(stream.read(chunk), np.float32))
  stream.stop_stream()
  stream.close()
  print("[*] Recording done")

  # Process data
  audio_data.resize((sample_rate * duration,))
  spec = librosa.feature.melspectrogram(y = audio_data, sr = sample_rate)
  spec = librosa.power_to_db(spec, ref = np.max)
  spec = (spec - spec.mean()) / spec.std()
  spec = (spec - spec.min()) / (spec.max() - spec.min())
  spec = spec.reshape((1,) + spec.shape + (1,))

  # Compute predictions
  gender = model_gender.predict(spec)[0]
  age = model_age.predict(spec)[0]

  # Interpret results
  print("[#] Gender:\t" + ("m" if np.argmax(gender) else "f"))
  print("[#] Age:\t" + str(age[0]))
  button.setText(("Male, " if np.argmax(gender) else "Female, ") + str(round(age[0])))
  app.processEvents()

def close():
  global pa

  # Close streams
  pa.terminate()

# Load models from HDF5
print("[*] Loading models from HDF5")
model_gender = load_model(model_path + "gender.h5")
model_age = load_model(model_path + "age.h5")

# Open audio stream from mic
print("[*] Connecting to microphone")
pa = pyaudio.PyAudio()

# Start GUI
app = QApplication(sys.argv)
button = QPushButton("Click to start recording")
app.setStyleSheet("* { font-size: 36pt; }")
button.clicked.connect(predict)
app.aboutToQuit.connect(close)
button.show()
app.exec()