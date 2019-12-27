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


def Preprocess(audio, clip=False):
    '''
    预处理，剪掉开头的一秒
    :param audio: 音频数组
    :param clip: 是否为片段
    :return:
    '''
    if not clip:
        data = audio[SAMPLE_FREQUENCY:]
    else:
        data = audio
    t = np.linspace(0, data.shape[0] / SAMPLE_FREQUENCY, data.shape[0] + 1)
    t = t[:-1]
    return data, t


def GetBaseBand(x, t, bias=0):
    '''
    计算乘以cos和sin后滤波得到的基带信号
    :param x: 数据
    :param t: 时间轴采样点
    :param bias: 用来做乘法的cos和sin信号的相位的偏移
    :return:
    '''
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
    '''
    根据正弦和余弦的基带信号得到的复数x计算相位角
    :param x: cos(phi) + isin(phi)
    :return:
    '''
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
    '''
    带通滤波器，用于初始滤波
    :param x: 原始数据值
    :return:
    '''
    b, a = signal.butter(8, [2 * FLOOR / SAMPLE_FREQUENCY, 2 * CEIL / SAMPLE_FREQUENCY], 'bandpass')
    filted = signal.filtfilt(b, a, x)
    return filted


def LowFilter(x):
    '''
    低通滤波器，用于滤波得到基带
    :param x: 相乘后的数据
    :return:
    '''
    b, a = signal.butter(6, 2 * LOWPASS_FREQUENCY / SAMPLE_FREQUENCY, 'lowpass')
    filted = signal.filtfilt(b, a, x)
    return filted


def PlotAudio(audio):
    '''
    将最终录制得到的信号作为整体可视化
    :param audio: 整段信号
    :return:
    '''
    data, t = Preprocess(audio)
    xr, xi = GetBaseBand(data, t)
    phi = ComputePhase(xr + np.complex(0, 1) * xi)
    d = CalculateDistance(phi)
    for i in range(NUMBER_OF_SIGNALS):
        plt.plot(d[i])
        plt.show()
    return d


def AnalyzeAudio(audio, bias=0):
    '''
    对录制的信号片段计算当前的距离
    :param audio: 信号片段
    :param bias: 相位偏移
    :return:
    '''
    data, t = Preprocess(audio, clip=True)
    xr, xi = GetBaseBand(data, t, bias)
    phi = ComputePhase(xr + np.complex(0, 1) * xi)
    d = CalculateDistance(phi)
    return d


def main():
    _, audio = wavfile.read('recv.wav')
    PlotAudio(audio)


if __name__ == '__main__':
    main()
