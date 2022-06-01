import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tqdm import tqdm

feature_path = "data/processed/"
model_path = "models/"
test_size = 0.1
random_state = 0
batch = 8
epochs = 30
val_size = 0.1
X, y_gender, y_age = [], [], []

if not len(os.listdir(model_path)):
  # Load speakers.csv
  speakers = pd.read_csv(feature_path + "speakers.csv", index_col = 0)

  # Load features from NumPy caches
  print("[*] Loading features from NumPy caches")
  for id in tqdm(speakers.index):
    for feature_file in glob(feature_path + str(id) + "/*.npy"):
      X.append(np.load(feature_file))
      y_gender.append(speakers["Gender"][id] == "m")
      y_age.append(int(speakers["Age"][id]))

  # Prepare features and targets
  X, y_gender, y_age = np.array(X), tf.keras.utils.to_categorical(y_gender, num_classes = len(set(y_gender))), np.array(y_age)
  X = X.reshape(X.shape + (1,))
  X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(X, y_gender, y_age, test_size = 0.1, random_state = 0)

  # Make and fit gender classification model
  print("[*] Training gender classification model")
  model_gender = Sequential([
    Input(X.shape[1:]),
    Conv2D(16, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(64, activation = "relu"),
    Dense(2, activation = "softmax")
  ])
  model_gender.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
  history_gender = model_gender.fit(X_train, y_gender_train, batch_size = batch, epochs = epochs, validation_split = val_size)

  # Make and fit age regression model
  print("[*] Training age regression model")
  model_age = Sequential([
    Input(X.shape[1:]),
    Conv2D(16, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(64, activation = "relu"),
    Dense(1)
  ])
  model_age.compile(loss = "mean_absolute_error", optimizer = "adam", metrics = ["accuracy"])
  history_age = model_age.fit(X_train, y_age_train, batch_size = batch, epochs = epochs, validation_split = val_size)

  # Display accuracy metrics for gender classification model
  plt.plot(epochs, history_gender.history["acc"], "r", label = "Training accuracy")
  plt.plot(epochs, history_gender.history["val_acc"], "b", label = "Validation accuracy")
  plt.title("Gender accuracy")
  plt.legend(loc = 0)
  plt.figure()
  plt.show()

  loss_gender, acc_gender = model_gender.evaluate(X_test, y_gender_test)

  print("[#] Gender training accuracy:\t" + str(history_gender.history["acc"]))
  print("[#] Gender testing accuracy:\t" + str(acc_gender))
  print("[#] Gender validation accuracy:\t" + str(history_gender.history["val_acc"]))

  # Display accuracy metrics for age regression model
  plt.plot(epochs, history_age.history["acc"], "r", label = "Training accuracy")
  plt.plot(epochs, history_age.history["val_acc"], "b", label = "Validation accuracy")
  plt.title("Age accuracy")
  plt.legend(loc = 0)
  plt.figure()
  plt.show()

  loss_age, acc_age = model_age.evaluate(X_test, y_age_test)

  print("[#] Age training accuracy:\t" + str(history_age.history["acc"]))
  print("[#] Age testing accuracy:\t" + str(acc_age))
  print("[#] Age validation accuracy:\t" + str(history_age.history["val_acc"]))

  # Save models as HDF5
  model_gender.save(model_path + "gender.h5")
  model_age.save(model_path + "age.h5")
  print("[*] Models saved")

else:
  print("[-] Models already trained, skipping")
