import numpy as np
from scipy.io import wavfile
import math
from scipy.fftpack import fft
import matplotlib.pyplot as plt

T_FREQUENCY = 0.04
MIN_FREQUENCY = 18000
MAX_FREQUENCY = 20500
SAMPLE_FREQUENCY = 48000
NUMBER_OF_SAMPLES = int(T_FREQUENCY * SAMPLE_FREQUENCY) + 1
WAVE_AMPLIFY_RATIO = 10000.0
NUMBER_OF_CHIRPS = 588


def LinearChirpSignal(t, T, fmin, fmax):
    '''
    :param t: 时间采样点，为一个np.ndarray
    :param T: 频率变换周期
    :param fmin: 最小频率
    :param fmax: 最大频率
    :return: 线性啁啾信号
    '''
    bandwidth = fmax - fmin
    data = [math.cos(2 * math.pi * (fmin + bandwidth / (2 * T) * x) * x) for x in t]
    data = np.array(data)
    return data


def GetChirpClip():
    t = np.linspace(0, T_FREQUENCY, NUMBER_OF_SAMPLES)
    chirp = LinearChirpSignal(t, T_FREQUENCY, MIN_FREQUENCY, MAX_FREQUENCY)
    return chirp


def GenerateSampleSignal(num_chirp):
    chirp = GetChirpClip()
    data = np.array([])
    silence = np.zeros(NUMBER_OF_SAMPLES)
    for i in range(num_chirp):
        data = np.concatenate((data, chirp, silence))
    return data


def GenerateAmplifiedWave(filename, data):
    output = data * WAVE_AMPLIFY_RATIO
    output = output.astype(np.short)
    wavfile.write(filename, SAMPLE_FREQUENCY, output)


def main():
    GenerateAmplifiedWave('highchirp.wav', GenerateSampleSignal(NUMBER_OF_CHIRPS))


if __name__ == '__main__':
    main()
