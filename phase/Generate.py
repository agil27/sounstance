import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile

SAMPLE_FREQUENCY = 48000
FREQUENCY_INTERVAL = 350
NUMBER_OF_SIGNALS = 1
BASE_FREQUENCY = 4500
DURATION = 300
AMPLITUDE = 10000.0

# 总共发射10个信号，频率从45000Hz开始，每次增加350Hz，FREQ即为得到的票i年率列表
FREQ = np.linspace(BASE_FREQUENCY,
                   BASE_FREQUENCY + (NUMBER_OF_SIGNALS - 1) * FREQUENCY_INTERVAL,
                   NUMBER_OF_SIGNALS)
# 时域上持续300秒，采样率为48000Hz
T = np.linspace(0, DURATION, SAMPLE_FREQUENCY * DURATION + 1)[:-1]

def GenerateSignal():
    freq = FREQ.copy()
    freq = freq.reshape(1, -1)
    t = T.copy()
    t = t.reshape(1, -1)
    # 多频率信号叠加
    signal = np.sum(AMPLITUDE * np.cos(2 * np.pi * np.matmul(freq.T, t)), axis=0) / NUMBER_OF_SIGNALS
    return signal, t


def displayFFTResult(t, signal, fs):
    freqs = np.linspace(0, fs // 2, len(t) // 2)
    fftRes = fft(signal)
    fftRes = fftRes[:(len(fftRes) // 2)]
    fftRes = fftRes.real
    # 傅里叶变换展现多频叠加的效果
    plt.plot(fftRes, freqs)
    plt.show()


def GenerateWavefile(data, filename):
    data = data.astype(np.short)
    wavfile.write(filename, SAMPLE_FREQUENCY, data)


def main():
    signal, t = GenerateSignal()
    t = t.reshape(-1)
    displayFFTResult(t, signal, SAMPLE_FREQUENCY)
    GenerateWavefile(signal, 'phase.wav')


if __name__ == '__main__':
    main()
