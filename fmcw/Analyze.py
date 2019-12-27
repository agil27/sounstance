from Generate import *
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt

CEIL = 23000
FLOOR = 17000
FFT_ZERO_APPEND_LENGTH = 1024 * 128
SOUND_VELOCITY = 340
THRESHOLD = 5000
LOWPASS_FREQUENCY = 5000


def ComputeStartPosition(data):
    '''
    利用Chirp信号和录制信号做乘积找最大值位置的方法来寻找录制信号的起始位置
    :param data: 录制信号
    :return:
    '''
    chirp = GetChirpClip()
    maxConvolution, maxPosition = 0, 0
    for i in range(NUMBER_OF_SAMPLES):
        convolution = chirp * data[i: i + NUMBER_OF_SAMPLES]
        convolution = convolution.sum()
        if convolution > maxConvolution:
            maxConvolution, maxPosition = convolution, i
    return maxPosition


def PseudoTransmittedSignal(num):
    return GenerateSampleSignal(num)


def Filter(x):
    '''
    简单的Butterworth带通滤波器
    :param x:
    :return:
    '''
    b, a = signal.butter(8, [2 * FLOOR / SAMPLE_FREQUENCY, 2 * CEIL / SAMPLE_FREQUENCY], 'bandpass')
    filted = signal.filtfilt(b, a, x)
    return filted


def ComputeFrequencyBias(start, step, pseudo, filted, num_chirps):
    '''
    利用FMCW的方法，通过FFT计算频率偏差
    :param start: 信号的开始位置
    :param step: 信号的窗口大小
    :param pseudo: Chirp信号
    :param filted: 滤波后的接收信号
    :param num_chirps: Chirp信号的个数
    :return: 频率的偏差构成的数组
    '''
    product = pseudo * filted
    indices = np.zeros(num_chirps)
    for i in range(
            start,
            start + step * (num_chirps - 1) + 1,
            step):
        fftResult = fft(product[i:(i + NUMBER_OF_SAMPLES + 1)], FFT_ZERO_APPEND_LENGTH)
        fftResult = np.abs(fftResult)
        fftResult = np.abs(fftResult[:int(FFT_ZERO_APPEND_LENGTH / 2)])
        index = np.argmax(fftResult)
        indices[(i - start) // step] = index
    return indices


def LowFilter(x):
    '''
    低通滤波器
    :param x: 信号
    :return:
    '''
    b, a = signal.butter(6, 2 * LOWPASS_FREQUENCY / SAMPLE_FREQUENCY, 'lowpass')
    filted = signal.filtfilt(b, a, x)
    return filted


def ComputeDistance(received, startIndex, startPosition):
    '''
    根据信号片段计算距离
    :param received: 接收到信号片段
    :param startIndex: 起始的Index，一般是0
    :param startPosition: 信号起始位置
    :return:
    '''
    numberOfChirps = (received.shape[0] - startPosition) // (2 * NUMBER_OF_SAMPLES)
    if numberOfChirps <= 0:
        return
    pseudo = PseudoTransmittedSignal(numberOfChirps)
    initialBias = np.zeros(startPosition)
    pseudo = np.concatenate((initialBias, pseudo))
    filted = Filter(received)
    n = filted.shape[0]
    m = pseudo.shape[0]
    pseudo = np.concatenate((pseudo, np.zeros(n - m)))
    pseudo = 10000 * pseudo
    chirpSilenceLength = NUMBER_OF_SAMPLES * 2
    indices = ComputeFrequencyBias(startPosition, chirpSilenceLength, pseudo, filted, numberOfChirps)
    distanceBias = (indices - startIndex) * SAMPLE_FREQUENCY * SOUND_VELOCITY * T_FREQUENCY
    distanceBias = distanceBias / FFT_ZERO_APPEND_LENGTH / (MAX_FREQUENCY - MIN_FREQUENCY)
    return distanceBias


def PlotDistance(received, startIndex):
    '''
    对整段信号绘制出距离的变化
    :param received: 接收到的信号
    :param startIndex: 起始Index，一般为0
    :return:
    '''
    received = received[SAMPLE_FREQUENCY:-SAMPLE_FREQUENCY]
    startPosition = ComputeStartPosition(received)
    if startPosition is None:
        return
    numberOfChirps = (received.shape[0] - startPosition) // (2 * NUMBER_OF_SAMPLES)
    pseudo = PseudoTransmittedSignal(numberOfChirps)
    initialBias = np.zeros(startPosition)
    pseudo = np.concatenate((initialBias, pseudo))
    filted = Filter(received)
    n = filted.shape[0]
    m = pseudo.shape[0]
    pseudo = np.concatenate((pseudo, np.zeros(n - m)))
    pseudo = 10000 * pseudo
    # 可视化用，用来展示信号的偏移
    """
    for i in range(numberOfChirps):
        plt.figure()
        plt.plot(pseudo[i * 7680:(i + 1) * 7680], color='red', linestyle='--')
        plt.plot(filted[i * 7680:(i + 1) * 7680], color='yellow', linestyle='-')
        plt.show()
    """
    chirpSilenceLength = NUMBER_OF_SAMPLES * 2
    indices = ComputeFrequencyBias(startPosition, chirpSilenceLength, pseudo, filted, numberOfChirps)
    distanceBias = (indices - startIndex) * SAMPLE_FREQUENCY * SOUND_VELOCITY * T_FREQUENCY
    distanceBias = distanceBias / FFT_ZERO_APPEND_LENGTH / (MAX_FREQUENCY - MIN_FREQUENCY)
    distanceBias = LowFilter(distanceBias)
    distanceBias = distanceBias - distanceBias[0]
    plt.plot(distanceBias)
    plt.show()
    return distanceBias
