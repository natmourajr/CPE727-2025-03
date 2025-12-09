"""
Analysis description Module

This module contains a class and functions for applying spectral analysis to input data.
"""
import enum
import typing

import numpy as np
import scipy.signal as sci
import librosa


class SpectralAnalysis(enum.Enum):
    """ Enum class to represent and process the available spectral analyzes in this module """
    SPECTROGRAM = 1
    LOG_SPECTROGRAM = 2
    LOFAR = 3
    LOFARGRAM = 3
    LOG_MELGRAM = 4

    def __str__(self):
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

    def apply(self, *args, **kwargs):
        """Perform data spectral analysis

        Args:
            - data (np.array): Input data for analysis.
            - fs (float): Sampling frequency.
            - n_pts (int, optional): Number of points in the FFT analysis. Defaults to 1024.
            - n_overlap (int, optional): Number of samples to overlap when splitting data into
                windows. Defaults to 0.
            - decimation_rate (int, optional): Decimation rate for downsampling the data.
                Defaults to 1.
            - n_mels (int, optional): Number of Mel points, applicable only in Mel analysis.
                Defaults to 256.

        Returns:
            typing.Tuple[np.array, np.array, np.array]: A tuple containing:
                - 2D array representing power spectrum.
                - 1D array with output frequencies.
                - 1D array with relative time to sample 0 of the data.

        """
        power, freq, time = globals()[str(self)](*args, **kwargs)

        integration_overlap = kwargs.get('integration_overlap', 0)
        integration_interval = kwargs.get('integration_interval', None)

        if integration_interval is None:
            return power, freq, time

        delta_t = time[-1] - time[-2]
        n_means = int(np.round(integration_interval / delta_t))
        n_overlap = int(np.round((integration_interval-integration_overlap)/ delta_t))

        final_power = []
        final_times = []

        for i in range(0, len(time), n_overlap):
            mean_spectrum = np.mean(power[:, i:i+n_means], axis=1)
            final_power.append(mean_spectrum)
            final_times.append(time[i])

        return np.array(final_power).T, freq, final_times

class Normalization(enum.Enum):
    """ Enum class representing the available normalizations in this module. """
    MIN_MAX = 0
    MIN_MAX_ZERO_CENTERED = 1
    NORM_L1 = 2
    NORM_L2 = 3
    NONE = 4

    def apply(self, data: np.array) -> np.array:
        """
        Apply normalization to input data. Equivalent to __cal__

        Args:
            data (np.array): The data to be normalized.

        Raises:
            UnboundLocalError: Raised when the specific normalization method is not
                implemented in this module.

        Returns:
            np.array: The normalized data.
        """
        if self == Normalization.MIN_MAX:
            return (data - np.min(data, axis=0))/(np.max(data, axis=0) - np.min(data, axis=0))

        if self == Normalization.MIN_MAX_ZERO_CENTERED:
            return data/np.max(np.abs(data), axis=0)

        if self == Normalization.NORM_L1:
            return data/np.linalg.norm(data, ord=1, axis=0)

        if self == Normalization.NORM_L2:
            data = Normalization.MIN_MAX.apply(data)
            return data/np.linalg.norm(data, ord=2, axis=0)

        if self == Normalization.NONE:
            return data

        raise UnboundLocalError(f"normalization {type:d} not implemented")

    def __call__(self, data: np.array) -> np.array:
        return self.apply(data)


def decimate(data: np.array, rate: typing.Union[int, float]):
    b, a = sci.cheby1(8, 0.05, 0.8 / rate, btype='low')
    y = sci.filtfilt(b, a, data)
    return sci.resample(y, int(len(y) / rate))

def tpsw(data: np.array, n_pts: int = None, n: int = None, p: int = None,
         a: int = None) -> np.array:
    """Perform TPSW data analysis

    Args:
        data (np.array): data to process
        n_pts (int, optional): number of samples in data to evaluate. Defaults to None(all samples).
        n (int, optional): TPSW n parameter, number of ones sample on each side zero filter.
            Defaults to None(int(round(n_pts*.04/2.0+1))).
        p (int, optional): TPSW p parameter, number of zeros from central sample.
            Defaults to None(int(round(n / 8.0 + 1)))
        a (int, optional): TPSW a parameter, threshold to saturate in the first pass filter.
            Defaults to None(2.0).

    Returns:
        np.array: processed data
    """

    if data.ndim == 1:
        data = data[:, np.newaxis]
    if n_pts is None:
        n_pts = data.shape[0]
    else:
        data = data[:n_pts, :]
    if n is None:
        n = int(round(n_pts * .04 / 2.0 + 1))
    if p is None:
        p = int(round(n / 8.0 + 1))
    if a is None:
        a = 2.0
    if p>0:
        h = np.concatenate((np.ones((n - p + 1)), np.zeros(2 * p - 1), np.ones((n - p + 1))))
    else:
        h = np.ones((1, 2 * n + 1))
        p = 1

    h /= np.linalg.norm(h, 1)
    def apply_on_spectre(xs):
        return sci.convolve(h, xs, mode='full')
    mx = np.apply_along_axis(apply_on_spectre, arr=data, axis=0)

    ix = int(np.floor((h.shape[0] + 1)/2.0))
    mx = mx[ix-1:n_pts+ix-1]
    ixp = ix - p
    mult = 2 * ixp / \
        np.concatenate([np.ones(p-1) * ixp, range(ixp,2*ixp + 1)], axis=0)[:, np.newaxis]
    mx[:ix,:] = mx[:ix,:] * (np.matmul(mult, np.ones((1, data.shape[1]))))
    mx[n_pts-ix:n_pts,:] = mx[n_pts-ix:n_pts,:] * \
        np.matmul(np.flipud(mult),np.ones((1, data.shape[1])))

    indl = (data-a*mx) > 0
    data = np.where(indl, mx, data)
    mx = np.apply_along_axis(apply_on_spectre, arr=data, axis=0)
    mx = mx[ix-1:n_pts+ix-1,:]
    mx[:ix,:] = mx[:ix,:] * (np.matmul(mult,np.ones((1, data.shape[1]))))
    mx[n_pts-ix:n_pts,:] = mx[n_pts-ix:n_pts,:] * \
        (np.matmul(np.flipud(mult),np.ones((1,data.shape[1]))))
    return mx

def spectrogram(data: np.array, fs: float, n_pts: int =1024, n_overlap: int =0,
        decimation_rate: int = 1, **kwargs) -> typing.Tuple[np.array, np.array, np.array]:
    """Perform spectrogram data analysis

        Args:
            data (np.array): Input data for analysis
            fs (float): Sampling frequency
            n_pts (int, optional): Number of points in the FFT analysis. Defaults to 1024.
            n_overlap (int, optional): Number of samples to overlap when splitting data into
                windows. Defaults to 0.
            decimation_rate (int, optional): _description_. Defaults to 1.

        Returns:
            typing.Tuple[np.array, np.array, np.array]: A tuple containing:
                - 2D array representing power spectrum.
                - 1D array with output frequencies.
                - 1D array with relative time to sample 0 of the data.
    """
    # pylint: disable=unused-argument

    data = data - np.mean(data)

    n_pts = n_pts * 2
    if n_overlap < 1:
        n_overlap = np.floor(n_pts * n_overlap)
    else:
        n_overlap = n_overlap * 2

    data = decimate(data, decimation_rate)
    fs = fs/decimation_rate

    freq, time, power = sci.spectrogram(data,
                                    nfft=n_pts,
                                    fs=fs,
                                    window=np.hanning(n_pts),
                                    noverlap=n_overlap,
                                    detrend=False,
                                    scaling='spectrum',
                                    mode='complex')
    power = np.abs(power)*n_pts/2
    power = power[1:,:]
    freq = freq[1:]
    return power, freq, time

def log_spectrogram(data: np.array, fs: float, n_pts: int =1024, n_overlap: int =0,
        decimation_rate: int = 1, **kwargs) -> typing.Tuple[np.array, np.array, np.array]:
    """Perform log_spectrogram data analysis

        Args:
            data (np.array): Input data for analysis
            fs (float): Sampling frequency
            n_pts (int, optional): Number of points in the FFT analysis. Defaults to 1024.
            n_overlap (int, optional): Number of samples to overlap when splitting data into
                windows. Defaults to 0.
            decimation_rate (int, optional): _description_. Defaults to 1.

        Returns:
            typing.Tuple[np.array, np.array, np.array]: A tuple containing:
                - 2D array representing power spectrum.
                - 1D array with output frequencies.
                - 1D array with relative time to sample 0 of the data.
    """
    # pylint: disable=unused-argument
    power, freq, time = spectrogram(data, fs, n_pts, n_overlap, decimation_rate)
    power[power < 1e-9] = 1e-9
    power = 20*np.log10(power)

    return power, freq, time

def lofar(data: np.array, fs: float, n_pts: int =1024, n_overlap: int =0,
        decimation_rate: int = 1, **kwargs) -> typing.Tuple[np.array, np.array, np.array]:
    """Perform lofar data analysis

        Args:
            data (np.array): Input data for analysis
            fs (float): Sampling frequency
            n_pts (int, optional): Number of points in the FFT analysis. Defaults to 1024.
            n_overlap (int, optional): Number of samples to overlap when splitting data into
                windows. Defaults to 0.
            decimation_rate (int, optional): _description_. Defaults to 1.

        Returns:
            typing.Tuple[np.array, np.array, np.array]: A tuple containing:
                - 2D array representing power spectrum.
                - 1D array with output frequencies.
                - 1D array with relative time to sample 0 of the data.
    """
    # pylint: disable=unused-argument
    power, freq, time = log_spectrogram(data, fs, n_pts, n_overlap, decimation_rate)
    power = power - tpsw(power)
    power[power < -0.2] = 0
    return power, freq, time

def log_melgram(data: np.array, fs: float, n_pts: int =1024, n_overlap: int =0, n_mels: int = 256,
        decimation_rate: int = 1, normalization: Normalization = Normalization.MIN_MAX_ZERO_CENTERED, **kwargs) -> typing.Tuple[np.array, np.array, np.array]:
    """Perform log_melgram data analysis

        Args:
            data (np.array): Input data for analysis
            fs (float): Sampling frequency
            n_pts (int, optional): Number of points in the FFT analysis. Defaults to 1024.
            n_overlap (int, optional): Number of samples to overlap when splitting data into
                windows. Defaults to 0.
            n_mels (int, optional): Number of Mel points, applicable only in Mel analysis.
                Defaults to 256.
            decimation_rate (int, optional): _description_. Defaults to 1.

        Returns:
            typing.Tuple[np.array, np.array, np.array]: A tuple containing:
                - 2D array representing power spectrum.
                - 1D array with output frequencies.
                - 1D array with relative time to sample 0 of the data.
        """
    n_fft=n_pts*2
    n_overlap *= 2
    hop_length=n_fft-n_overlap
    discard=int(np.floor(n_fft/hop_length))

    if decimation_rate != 1:
        data = decimate(data, decimation_rate)
        fs = fs/decimation_rate

    fmax=fs/2
    n_data = normalization(data).astype(float)
    power = librosa.feature.melspectrogram(
                    y=n_data,
                    sr=fs,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window=np.hanning(n_fft),
                    n_mels=n_mels,
                    power=2,
                    fmax=fmax)
    power = librosa.power_to_db(power, ref=np.max)
    power = power[:,discard:]

    freqs = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fmax)

    start_time = n_pts/fs
    step_time = (n_fft-n_overlap)/fs
    times = [start_time + step_time * valor for valor in range(power.shape[1])]
    return power, freqs, times
