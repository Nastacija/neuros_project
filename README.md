Пайплайн представляет собой цепочку нейросетей. Первой нейросети на вход подается аудиофайл, на выходе - список позиций, на которых должны находиться метки. Аудиофайл и список меток подается на вход второй нейросети, которая предсказывает класс для каждого сегмента.

1. Настройка пайплайна

requirements.txt - установить окружение

config.py - прописать необходимые переменные (классы фонем; путь к датасету для обучения; путь к аудиофайлу и его первичная обработка)

2. Подготовка данных

cut_audio.py - скрипт, при помощи которого был получен обучающий датасет (200 файлов corpress были разрезаны на слова)

prepare_audiodata.py - подготовка данных для обучения первой нейросети (предсказатель позиций меток)

prepare_phonem_data.py - подготовка данных для обучения второй нейросети (классификатор)

3. Обучение моделей

classes_training.py - обучение модели-предсказателя

model1_checkpoint.weights.h5 - сохраняем веса

labels_training.py - обучение модели-классификатора

model2_checkpoint.weights.h5 - сохраняем веса

4. Запуск пайплайна

seg_output_demo.py - файл запуска пайплайна; на выходе - сег-файл с метками уровня В1 и классами фонем (сейчас в нем тестовые данные)

labels_prediction.py, classes_prediction.py - нейросети

5. Утилиты

read_write_seg.py - скрипт для чтения и записи сег-файлов (две версии записи сег-файлов: на основе словаря меток и на основе списка позиций и списка имен)

sbl2wav.py - скрипт для конвертации .sbl-файлов в .wav-файлы

6. Пример входных и выходнях данных пайплайна

ata_0_0002.wav - входные данные

ata_0_0002.seg - выходные данные
