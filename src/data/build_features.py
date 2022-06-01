import os
import tqdm
import librosa
import numpy as np
import pandas as pd
from glob import glob

dataset_path = glob("data/external/*/")[0]
feature_path = "data/processed/"
sample_rate = 48000
duration = 3
spectrograms = []

# Check features existence
if not len(os.listdir(feature_path)):
  print("[*] Patching speakers.csv")
  # Import speakers metadata
  speakers = pd.read_csv(dataset_path + "speakers.csv")

  # speakers.csv is padded with redundant lines, remove
  speakers.dropna(axis = 0, how = "all", inplace = True)
  speakers.dropna(axis = 1, how = "all", inplace = True)

  # Convert ID to int and set as index
  speakers.set_index(speakers["ID"].astype(int), inplace = True)
  speakers.drop("ID", axis = 1, inplace = True)

  # Save modified metadata
  speakers.to_csv(feature_path + "speakers.csv")

  print("[*] Processing FLAC audio files")
  for id in tqdm(speakers.index):
    if not os.path.exists(feature_path + str(id)):
      os.mkdir(feature_path + str(id))

    for audio_file in glob(dataset_path + "*/" + str(id) + "/*.flac"):
      # Generate log mel spectrogram
      audio = librosa.load(audio_file, sr = sample_rate, duration = duration)[0].copy()
      audio.resize(sample_rate * duration)
      spec = librosa.feature.melspectrogram(y = audio, sr = sample_rate)
      spec = librosa.power_to_db(spec, ref = np.max)
      spec = (spec - spec.mean()) / spec.std()
      spec = (spec - spec.min()) / (spec.max() - spec.min())

      # Save as .npy
      np.save(feature_path + str(id) + "/" + "".join(audio_file.split("/")[-1].split(".")[:-1]) + ".npy", spec) # Glob FLAC files
else:
  print("[-] Features exist, skipping")