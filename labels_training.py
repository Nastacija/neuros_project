import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
from prepare_audiodata import x_train, x_test, y_train, y_test, max_len_aud

#прописываем модель
model1 = Sequential()
model1.add(Input(shape=(max_len_aud, 1)))
model1.add(Conv1D(32, 3, activation='relu', padding='same'))
model1.add(Conv1D(64, 3, activation='relu', padding='same'))
model1.add(LSTM(64, return_sequences=True))
model1.add(Dense(1, activation='sigmoid'))

#компилируем модель
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#обучаем модель
model1.fit(x_train[:500], y_train[:500], epochs=5, batch_size=25, verbose=1)
model1.save_weights(r'model1_checkpoint.weights.h5')

