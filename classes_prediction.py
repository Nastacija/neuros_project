import tensorflow as tf
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from labels_prediction import predicted_indices
from config import audio_input, fs

model2 = Sequential()
model2.add(Input(shape=(40,)))
model2.add(Dense(4, activation="softmax", use_bias=False))
model2.load_weights(r'model2_checkpoint.weights.h5')

#делим слово на фонемы по предсказанным индексам
phon_parts = []
for l1, l2 in zip(predicted_indices, predicted_indices[1:]):
    aud = audio_input[l1:l2+1]
    l = len(aud)
    begin = (l-512)//2
    end = begin + 512
    centre = np.array(aud[begin:end])
    phon_parts.append(centre)

#извлекаем mfcc для каждой предполагаемой фонемы
param_list = []
for phon in phon_parts:
    mfccs = librosa.feature.mfcc(y=phon, sr=fs, n_fft=512)
    if mfccs.shape == (20, 1):
        mfccs = np.append(mfccs, np.zeros((20, 1)), axis=0)
        mfccs = mfccs.reshape(20, 2)
    shp = mfccs.shape[0] * mfccs.shape[1]
    param_list.append(mfccs.reshape(shp,))

predicted_classes = []
for dt in param_list:
    phon_input = tf.expand_dims(dt.reshape(1,40), axis=-1)
    prediction = model2.predict(phon_input)
    output_class = np.argmax(prediction)
    predicted_classes.append(output_class)

print(predicted_classes)