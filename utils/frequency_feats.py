
import scipy.signal
from scipy.fftpack import fft
from scipy.stats import binned_statistic
from scipy import signal
import numpy as np


def fir_filter(data, cutoff, fs, order=3, highpass=False):
    nyq = 0.5 * fs
    cutoff_nyq = cutoff / nyq
    if highpass:
        b = scipy.signal.firwin(order, cutoff_nyq, pass_zero='highpass')
    else:
        b = scipy.signal.firwin(order, cutoff_nyq)

    y = scipy.signal.filtfilt(b, 1, data)
    return y


def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    vals = 2.0 / N * np.abs(yf[0:N // 2])  # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    return xf, vals


def freq_feats(audio, fs, new_fs=560):

    audio = fir_filter(audio, 280, fs, order=15, highpass=False)
    number_of_samples = round(len(audio) * float(new_fs) / fs)
    audio = signal.resample(audio, number_of_samples)

    x, spec = custom_fft(audio, new_fs)
    bin_means = binned_statistic(x, spec, bins=new_fs, )[0]

    return bin_means

