from Generate import *
from scipy.signal import convolve as conv
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft

CEIL = 10000
FLOOR = 2500

SEGMENT_SIZE = 32
SOUND_VELOCITY = 343
LOWPASS_FREQUENCY = 80

# CIC滤波器系数
CIC_DEC = 16
CIC_DELAY = 17
SECTION_N = 4

# LEVD系数
POWER_THR = 5.5e5
PEAK_THR = 220
DC_TREND = 0.25


def Preprocess(audio, clip=False):
    if not clip:
        data = audio[SAMPLE_FREQUENCY:]
    else:
        data = audio
    dataLength = data.shape[0] // (CIC_DEC * SEGMENT_SIZE)
    data = data[:dataLength * CIC_DEC * SEGMENT_SIZE]
    t = np.linspace(0, data.shape[0] / SAMPLE_FREQUENCY, data.shape[0] + 1)
    t = t[:-1]
    return data, t, dataLength


def GetBaseBand(x, t, bias=0):
    dataLength = x.shape[0]
    yr = np.zeros((NUMBER_OF_SIGNALS, dataLength))
    yi = np.zeros((NUMBER_OF_SIGNALS, dataLength))
    for i in range(NUMBER_OF_SIGNALS):
        xr = x * np.cos(2 * np.pi * FREQ[i] * (t + bias))
        xi = -x * np.sin(2 * np.pi * FREQ[i] * (t + bias))
        yr[i][:] = LowFilter(xr)
        yi[i][:] = LowFilter(xi)
    return yr, yi


def ComputePhase(x):
    phi = np.angle(x)
    for i in range(NUMBER_OF_SIGNALS):
        for j in range(1, x.shape[1]):
            k = int(phi[i][j - 1] / (2 * np.pi))
            cur = k * 2 * np.pi + phi[i][j]
            # 因为相位是单调变化的，所以需要加上2pi的整数倍保证不会发生2pi的相位跳动
            while cur > phi[i][j - 1] + np.pi:
                cur = cur - 2 * np.pi
            while cur < phi[i][j - 1] - np.pi:
                cur = cur + 2 * np.pi
            phi[i][j] = cur
    return phi


def CalculateDistance(phi):
    '''
    根据相位计算距离
    :param phi: 得到的相位向量，包含十个频率的相位信息
    :return: 距离（可能包含初始的误差，但变化量是相对准确的）
    '''
    distance = np.ndarray(phi.shape)
    for i in range(NUMBER_OF_SIGNALS):
        distance[i] = -phi[i] * SOUND_VELOCITY / 2 / np.pi / FREQ[i]
    return distance


def BandFilter(x):
    b, a = signal.butter(8, [2 * FLOOR / SAMPLE_FREQUENCY, 2 * CEIL / SAMPLE_FREQUENCY], 'bandpass')
    filted = signal.filtfilt(b, a, x)
    return filted


def LowFilter(x):
    b, a = signal.butter(6, 2 * LOWPASS_FREQUENCY / SAMPLE_FREQUENCY, 'lowpass')
    filted = signal.filtfilt(b, a, x)
    return filted


def PlotAudio(audio):
    data, t, dataLength = Preprocess(audio)
    xr, xi = GetBaseBand(data, t)
    phi = ComputePhase(xr + np.complex(0, 1) * xi)
    d = CalculateDistance(phi)
    for i in range(NUMBER_OF_SIGNALS):
        plt.plot(d[i])
        plt.show()
    return d


def AnalyzeAudio(audio, bias=0):
    data, t, dataLength = Preprocess(audio, clip=True)
    xr, xi = GetBaseBand(data, t, bias)
    phi = ComputePhase(xr + np.complex(0, 1) * xi)
    d = CalculateDistance(phi)
    return d


def main():
    _, audio = wavfile.read('recv.wav')
    PlotAudio(audio)


if __name__ == '__main__':
    main()
