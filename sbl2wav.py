import wave

def sbl_to_wav(filename: str, samplerate: int, sampwidth: int = 2, num_channels: int = 1) -> str:
    with open(filename, "rb") as fi:
        raw_signal = fi.read()

    res_filename = filename.split(".")[0] + ".wav"
    fo = wave.open(res_filename, "wb")  # открываем на запись (w) в бинарном режиме (b)
    fo.setnchannels(num_channels)
    fo.setsampwidth(sampwidth)
    fo.setframerate(samplerate)
    fo.writeframes(raw_signal)
    fo.close()
    return res_filename