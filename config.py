import glob
import parselmouth
from scipy.io.wavfile import read
import numpy as np

phon_classes = {
    'voiced': ["b", "b'", "v", "v'", "g", "g'", "d", "d'", "zh", "z", "z'"],
    'not_voiced': ["p", "p'", "f", "f'", "k", "k'", "t", "t'", "zh_", "s", "s'", "ch_", "ch", "sh", "sc"],
    'sonant': ["m", "m'", "n", "n'", "l", "l'", "r", "r'", "j"],
    'vowel': ["a1", "a2", "a4", "a0", "o1", "o2", "o4", "o0", "e1", "e2", "e4", "e0", "y1", "y2", "y4", "y0", "u1", "u2", "u4", "u0", "i1", "i2", "i4", "i0"]
}

files = list(glob.glob(r"C:\Users\Anastasiya\Python_tasks\neuros_project\dataset\*.seg_B1"))

#путь к записи
audio = 'ata_0_0002.wav'
fs, data = read(audio)
sound = parselmouth.Sound(values=data, sampling_frequency=fs)
channel = np.asarray(sound, dtype=np.float32)
shp = channel.shape
audio_input = channel.reshape(shp[1])