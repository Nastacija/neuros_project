import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
from prepare_audiodata import x_train, x_test, y_train, y_test, max_len_aud
from config import audio_input


model1 = Sequential()
model1.add(Input(shape=(max_len_aud, 1)))
model1.add(Conv1D(32, 3, activation='relu', padding='same'))
model1.add(Conv1D(64, 3, activation='relu', padding='same'))
model1.add(LSTM(64, return_sequences=True))
model1.add(Dense(1, activation='sigmoid'))
model1.load_weights(r'model1_checkpoint.weights.h5')

prediction = model1.predict(audio_input)
predicted_indices = np.where(prediction > 0.4)[0]
print(predicted_indices)