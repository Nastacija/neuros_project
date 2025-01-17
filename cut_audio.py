from sbl2wav import sbl_to_wav
from read_write_seg import read_seg, write_seg_light
import parselmouth
import numpy as np
import copy
import glob
from scipy.io.wavfile import read, write

def cut_audio(seg_file_Y, directory):
    # читаем сег со словами
    ind = seg_file_Y[-11:-7]
    params_Y, labels_Y = read_seg(seg_file_Y, encoding="cp1251")
    filename = seg_file_Y.rstrip('seg_Y1')

    # читаем сег с фонемами
    seg_file_B = f"{filename}seg_B1"
    params_B, labels_B = read_seg(seg_file_B)

    # читаем звук
    sbl_file = f"{filename}sbl"
    wav_file = sbl_to_wav(sbl_file, params_Y['SAMPLING_FREQ'])
    fs, data = read(wav_file)
    sound = parselmouth.Sound(values=data, sampling_frequency=fs)
    channel = np.asarray(sound, dtype=np.int16)
    shp = channel.shape
    channel = channel.reshape(shp[1])

    #записываем позиции меток
    pos_Y = []
    pos_B = []
    for label in labels_Y:
        pos_Y.append(label['position'])
    for label in labels_B:
        pos_B.append(label)

    #делим аудио на слова, метки фонем на списки для каждого слова
    lbl_parts = []
    aud_parts = []
    for l1, l2 in zip(pos_Y, pos_Y[1:]):
        lbl_lst = []
        for lbl in pos_B:
            if l1 <= lbl['position'] < l2:
                lbl_lst.append(lbl)
            if lbl['position'] == l2:
                temp = copy.deepcopy(lbl)
                temp['name'] = ''
                lbl_lst.append(temp)
        aud = channel[l1:l2+1]
        lbl_parts.append(lbl_lst)
        aud_parts.append(aud)

    #записываем куски аудио
    ctr = 0
    for part in aud_parts:
        write(f"{directory}_{ctr}_{ind}.wav", params_Y['SAMPLING_FREQ'], part)
        ctr += 1

    #корректируем позиции меток для сег-файла
    for lst in lbl_parts:
        value = lst[0]['position']
        for lbl in lst:
            lbl['position'] = lbl['position'] - value
    
    #записываем сег-файлы для каждого слова
    ctr = 0
    for part, lbls in zip(aud_parts, lbl_parts):
        write_seg_light(params_B, lbls, f"{directory}_{ctr}_{ind}.seg_B1", encoding = "utf-8-sig")
        ctr += 1

Y1_files = list(glob.glob(r"C:\Users\Anastasiya\Python_tasks\neuros_project\cta\*.seg_Y1"))
print(Y1_files)
for file in Y1_files:
    cut_audio(file, 'cta')
    print(file)
