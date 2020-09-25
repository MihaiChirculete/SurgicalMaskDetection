import threading
import librosa as lr
import matplotlib.pyplot as plt
import numpy as np


def audio_to_image(path, height=192, width=192):
    signal, sr = lr.load(path, res_type='kaiser_fast')
    hl = signal.shape[0] // (width * 1.1)
    spec = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hl))
    img = lr.power_to_db(spec) ** 2
    start = (img.shape[1] - width) // 2
    return img[:, start:start + width]
