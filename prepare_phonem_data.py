from read_write_seg import read_seg
from scipy.io.wavfile import read
import parselmouth
import librosa
import numpy as np
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from config import phon_classes
from config import files

def get_mfccs(seg_file):
    # читаем сег с фонемами
    params, labels = read_seg(seg_file)
    filename = seg_file.rstrip('seg_B1')

    # читаем звук
    wav_file = f'{filename}wav'
    fs, data = read(wav_file)
    sound = parselmouth.Sound(values=data, sampling_frequency=fs)
    channel = np.asarray(sound, dtype=np.float32)
    shp = channel.shape
    channel = channel.reshape(shp[1])

    #записываем позиции и имена меток
    phon_pos = []
    names = []
    for label in labels:
        phon_pos.append(label['position'])
        if label['name'] != '':
            names.append(label['name'])

    classes = []
    for nm in names:
        if nm in phon_classes['voiced']:
            classes.append(0)
        if nm in phon_classes['not_voiced']:
            classes.append(1)
        if nm in phon_classes['sonant']:
            classes.append(2)
        if nm in phon_classes['vowel']:
            classes.append(3)

    classes_ohe = utils.to_categorical(classes, 4)

    #делим слово на фонемы
    phon_parts = []
    for l1, l2 in zip(phon_pos, phon_pos[1:]):
        aud = channel[l1:l2+1]
        l = len(aud)
        begin = (l-512)//2
        end = begin + 512
        centre = np.array(aud[begin:end])
        phon_parts.append(centre)

    #извлекаем mfcc для каждой фонемы
    param_list = []
    for phon in phon_parts:
        mfccs = librosa.feature.mfcc(y=phon, sr=fs, n_fft=512)
        if mfccs.shape == (20, 1):
            mfccs = np.append(mfccs, np.zeros((20, 1)), axis=0)
            mfccs = mfccs.reshape(20, 2)
        shp = mfccs.shape[0] * mfccs.shape[1]
        param_list.append(mfccs.reshape(shp,))

    if len(classes_ohe) < len(param_list):
        n = len(param_list) - len(classes_ohe)

    if len(classes_ohe) < len(param_list):
        for i in range(0, n):
            classes_ohe = np.append(classes_ohe, np.zeros((1, 4)), axis=0)
    
    return classes_ohe, param_list

classes = []
mfccs = []

for fl in files:
    classes_ohe, param_list = get_mfccs(fl) 
    for data in param_list:
        mfccs.append(data)
    for data in classes_ohe:
        classes.append(data)

mfccs = np.asarray(mfccs)
classes = np.asarray(classes)

x_train_cl, x_test_cl, y_train_cl, y_test_cl = train_test_split(mfccs, classes, test_size=0.2)

print(mfccs[0])
print(classes[0])






