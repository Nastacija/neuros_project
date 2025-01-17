from read_write_seg import read_seg
from scipy.io.wavfile import read
import numpy as np
import parselmouth
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from config import files

def create_vecs(seg_file):
    # читаем сег с фонемами
    params, labels = read_seg(seg_file)
    filename = seg_file.rstrip('seg_B1')

    # читаем звук
    wav_file = f'{filename}wav'
    fs, data = read(wav_file)
    sound = parselmouth.Sound(values=data, sampling_frequency=fs)
    channel = np.asarray(sound, dtype=np.int16)
    shp = channel.shape
    channel = channel.reshape(shp[1])

    #записываем позиции меток
    pos = []
    for label in labels:
        pos.append(label['position'])
    print(pos)

    print(len(channel))
    binary_labels = np.zeros(len(channel))

    for lbl in pos:
        binary_labels[lbl] = 1
    
    return channel, binary_labels

#получаем список массивов для аудио и список массивов бинарного представления меток
audio_data = []
labels_data = []
for fl in files:
    channel, binary_labels = create_vecs(fl)
    audio_data.append(channel)
    labels_data.append(binary_labels)

#предобработка данных
def preprocess_data(audio, labels):
    max_len_aud = max(len(item) for item in audio)
    max_len_lbl = max(len(item) for item in labels)
    if max_len_aud == max_len_lbl:
        pad_audio = pad_sequences(audio, maxlen=max_len_aud, padding='post', dtype='int32')
        pad_audio = np.expand_dims(pad_audio, axis=-1)
        pad_labels = pad_sequences(labels, maxlen=max_len_lbl, padding='post', dtype='int32')
    return pad_audio, pad_labels, max_len_aud

pad_audio, pad_labels, max_len_aud = preprocess_data(audio_data, labels_data)

x_train, x_test, y_train, y_test = train_test_split(pad_audio, pad_labels, test_size=0.2)

print(files[:20])