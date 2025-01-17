import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from prepare_phonem_data import x_train_cl, x_test_cl, y_train_cl, y_test_cl
import numpy as np

model2 = Sequential()
model2.add(Input(shape=(40,)))
model2.add(Dense(400, activation="relu", use_bias=False))
model2.add(Dense(4, activation="softmax", use_bias=False))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(tf.expand_dims(x_train_cl, axis=-1), y_train_cl, batch_size=300, epochs=50)
model2.save_weights(r'model2_checkpoint.weights.h5')