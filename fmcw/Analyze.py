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
    b, a = signal.butter(8, [2 * FLOOR / SAMPLE_FREQUENCY, 2 * CEIL / SAMPLE_FREQUENCY], 'bandpass')
    filted = signal.filtfilt(b, a, x)
    return filted


def ComputeFrequencyBias(start, step, pseudo, filted, num_chirps):
    product = pseudo * filted
    print(len(product))
    product = LowFilter(product)
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
    b, a = signal.butter(6, 2 * LOWPASS_FREQUENCY / SAMPLE_FREQUENCY, 'lowpass')
    filted = signal.filtfilt(b, a, x)
    return filted


def ComputeDistance(received, startIndex, startPosition):
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
    chirpSilenceLength = NUMBER_OF_SAMPLES * 2
    indices = ComputeFrequencyBias(startPosition, chirpSilenceLength, pseudo, filted, numberOfChirps)
    distanceBias = (indices - startIndex) * SAMPLE_FREQUENCY * SOUND_VELOCITY * T_FREQUENCY
    distanceBias = distanceBias / FFT_ZERO_APPEND_LENGTH / (MAX_FREQUENCY - MIN_FREQUENCY)
    return distanceBias


def PlotDistance(received, startIndex):
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
    chirpSilenceLength = NUMBER_OF_SAMPLES * 2
    indices = ComputeFrequencyBias(startPosition, chirpSilenceLength, pseudo, filted, numberOfChirps)
    distanceBias = (indices - startIndex) * SAMPLE_FREQUENCY * SOUND_VELOCITY * T_FREQUENCY
    distanceBias = distanceBias / FFT_ZERO_APPEND_LENGTH / (MAX_FREQUENCY - MIN_FREQUENCY)
    distanceBias = LowFilter(distanceBias)
    distanceBias = distanceBias - distanceBias[0]
    t = np.linspace(0, len(filted) * 0.16, len(filted) // 7680)
    plt.plot(distanceBias)
    plt.show()
    return distanceBias
