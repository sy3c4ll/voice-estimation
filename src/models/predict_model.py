import librosa
import os
import sys
import numpy as np
from glob import glob
from tensorflow.keras.models import load_model

model_path = "models/"
target = sys.argv[1]
sample_rate = 48000
duration = 3

if os.path.exists(model_path + "gender.h5") and os.path.exists(model_path + "age.h5"):
  # Load models from HDF5
  print("[*] Loading models from HDF5")
  model_gender = load_model(model_path + "gender.h5")
  model_age = load_model(model_path + "age.h5")

  # Prepare target
  audio = librosa.load(target, sr = sample_rate, duration = duration)[0].copy()
  audio.resize(sample_rate * duration)
  spec = librosa.feature.melspectrogram(y = audio, sr = sample_rate)
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

else:
  print("[-] Models not found, skipping")
