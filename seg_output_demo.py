# from labels_prediction import predicted_indices
# from classes_prediction import predicted_classes
from config import audio, fs
from read_write_seg import write_seg_with_params

predicted_indices = [0, 3000, 5000, 6500, 10000, 12000]
predicted_classes = [3, 0, 2, 3, 2]
filename = audio.rstrip('wav')
filename_ext = f'{filename}seg'

names = []
for cls in predicted_classes:
    if cls == 0:
        names.append('voiced')
    if cls == 1:
        names.append('not_voiced')
    if cls == 2:
        names.append('sonant')
    if cls == 3:
        names.append('vowel')
names.append('')


write_seg_with_params(fs, predicted_indices, names, filename_ext, encoding = "utf-8-sig")