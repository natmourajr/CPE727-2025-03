"""
Feature extraction module for underwater acoustic signals.

Implements LOFAR (Low Frequency Analysis and Recording) and MEL spectrograms
for underwater acoustic target recognition tasks.

DEPRECATED: This module is deprecated. Use src.data.preprocessing.AudioPreprocessor instead,
which fully implements the paper's methodology (Section V-A) with correct parameters.
"""

import warnings

import numpy as np
import scipy.signal as signal
import librosa
from typing import Tuple, Optional
import torch


class TPSWFilter:
    """Two-Pass Split-Window filter for LOFAR processing."""

    @staticmethod
    def apply(data: np.ndarray, n_pts: Optional[int] = None,
              n: Optional[int] = None, p: Optional[int] = None,
              a: Optional[float] = None) -> np.ndarray:
        """
        Apply TPSW filtering to spectral data.

        Args:
            data: Input data (1D or 2D array)
            n_pts: Number of samples to process
            n: Number of ones on each side of zero filter
            p: Number of zeros from central sample
            a: Threshold for saturation in first pass

        Returns:
            Filtered data
        """
        if data.ndim == 1:
            data = data[:, np.newaxis]

        if n_pts is None:
            n_pts = data.shape[0]
        else:
            data = data[:n_pts, :]

        if n is None:
            n = int(round(n_pts * 0.04 / 2.0 + 1))
        if p is None:
            p = int(round(n / 8.0 + 1))
        if a is None:
            a = 2.0

        # Create filter
        if p > 0:
            h = np.concatenate([
                np.ones(n - p + 1),
                np.zeros(2 * p - 1),
                np.ones(n - p + 1)
            ])
        else:
            h = np.ones(2 * n + 1)
            p = 1

        h /= np.linalg.norm(h, 1)

        # Apply convolution
        def apply_on_spectre(xs):
            return signal.convolve(h, xs, mode='full')

        mx = np.apply_along_axis(apply_on_spectre, arr=data, axis=0)

        # Edge correction
        ix = int(np.floor((h.shape[0] + 1) / 2.0))
        mx = mx[ix-1:n_pts+ix-1]
        ixp = ix - p
        mult = 2 * ixp / np.concatenate([
            np.ones(p-1) * ixp,
            np.arange(ixp, 2*ixp + 1)
        ])[:, np.newaxis]

        # Apply edge correction with proper broadcasting
        if ix <= mx.shape[0]:
            mx[:ix, :] = mx[:ix, :] * mult
        if n_pts-ix >= 0:
            mx[n_pts-ix:n_pts, :] = mx[n_pts-ix:n_pts, :] * np.flipud(mult)

        # First pass
        indl = (data - a * mx) > 0
        data = np.where(indl, mx, data)

        # Second pass
        mx = np.apply_along_axis(apply_on_spectre, arr=data, axis=0)
        mx = mx[ix-1:n_pts+ix-1, :]

        # Apply edge correction with proper broadcasting
        if ix <= mx.shape[0]:
            mx[:ix, :] = mx[:ix, :] * mult
        if n_pts-ix >= 0:
            mx[n_pts-ix:n_pts, :] = mx[n_pts-ix:n_pts, :] * np.flipud(mult)

        return mx


class FeatureExtractor:
    """Base class for feature extraction.

    DEPRECATED: Use src.data.preprocessing.AudioPreprocessor instead.

    Note: Default parameters now match paper specification (Section V-A):
    - n_fft: 1024 (was 2048)
    - hop_length: 1024 (was 512, no overlap as per paper)
    """

    def __init__(self,
                 sr: int = 16000,
                 n_fft: int = 1024,  # Fixed to match paper (was 2048)
                 hop_length: int = 1024,  # Fixed to match paper (was 512, no overlap)
                 n_mels: Optional[int] = None):
        """
        Initialize feature extractor.

        Args:
            sr: Sample rate (Hz) - paper uses 16 kHz
            n_fft: FFT window size - paper uses 1024
            hop_length: Number of samples between successive frames - paper uses 1024 (no overlap)
            n_mels: Number of mel bands (for MEL spectrogram) - paper uses 128
        """
        warnings.warn(
            "FeatureExtractor is deprecated. Use src.data.preprocessing.AudioPreprocessor "
            "for paper-compliant implementation with averaging and L2 normalization.",
            DeprecationWarning,
            stacklevel=2
        )
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels or 128  # Changed from 256 to 128 to match paper

    def extract(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from audio signal.

        Args:
            audio: Input audio signal

        Returns:
            Tuple of (power, frequencies, times)
        """
        raise NotImplementedError


class LOFARExtractor(FeatureExtractor):
    """LOFAR (Low Frequency Analysis and Recording) spectrogram extractor.

    DEPRECATED: Use src.data.preprocessing.AudioPreprocessor instead.
    """

    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop_length: int = 1024):
        """
        Initialize LOFAR extractor.

        Args:
            sr: Sample rate (Hz) - paper uses 16 kHz
            n_fft: FFT window size - paper uses 1024
            hop_length: Number of samples between successive frames - paper uses 1024
        """
        super().__init__(sr, n_fft, hop_length)
        self.tpsw_filter = TPSWFilter()

    def extract(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract LOFAR features from audio signal.

        LOFAR = Log Spectrogram - TPSW(Log Spectrogram)
        This highlights consistent frequency patterns while minimizing transient noises.

        Args:
            audio: Input audio signal (1D array)

        Returns:
            Tuple of (power [freq x time], frequencies, times)
        """
        # Remove DC component
        audio = audio - np.mean(audio)

        # Compute spectrogram
        frequencies, times, Sxx = signal.spectrogram(
            audio,
            fs=self.sr,
            window=signal.get_window('hann', self.n_fft),
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            detrend=False,
            scaling='spectrum',
            mode='complex'
        )

        # Magnitude spectrum
        power = np.abs(Sxx) * self.n_fft / 2

        # Remove DC bin
        power = power[1:, :]
        frequencies = frequencies[1:]

        # Convert to log scale (with numerical stability)
        power[power < 1e-9] = 1e-9
        power_db = 20 * np.log10(power)

        # Apply TPSW normalization
        power_tpsw = self.tpsw_filter.apply(power_db)

        # LOFAR = log spectrogram - TPSW filtered version
        lofar = power_db - power_tpsw

        # Threshold negative values
        lofar[lofar < -0.2] = 0

        return lofar, frequencies, times

    def extract_torch(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract LOFAR features and return as PyTorch tensor.

        Args:
            audio: Input audio signal

        Returns:
            PyTorch tensor of shape [1, freq, time] (channel first for CNN)
        """
        lofar, _, _ = self.extract(audio)
        return torch.from_numpy(lofar).unsqueeze(0).float()


class MELExtractor(FeatureExtractor):
    """MEL spectrogram extractor.

    DEPRECATED: Use src.data.preprocessing.AudioPreprocessor instead.
    """

    def __init__(self, sr: int = 16000, n_fft: int = 1024,
                 hop_length: int = 1024, n_mels: int = 128):
        """
        Initialize MEL extractor.

        Args:
            sr: Sample rate (Hz) - paper uses 16 kHz
            n_fft: FFT window size - paper uses 1024
            hop_length: Number of samples between successive frames - paper uses 1024
            n_mels: Number of mel bands - paper uses 128
        """
        super().__init__(sr, n_fft, hop_length, n_mels)

    def extract(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract MEL spectrogram features from audio signal.

        MEL spectrogram maps frequencies to mel scale, which approximates
        human auditory perception.

        Args:
            audio: Input audio signal (1D array)

        Returns:
            Tuple of (power [mel_bins x time], mel_frequencies, times)
        """
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window='hann',
            n_mels=self.n_mels,
            power=2.0,
            fmax=self.sr / 2
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Get mel frequencies
        mel_freqs = librosa.mel_frequencies(n_mels=self.n_mels, fmin=0.0, fmax=self.sr/2)

        # Compute time axis
        times = librosa.frames_to_time(
            np.arange(mel_spec_db.shape[1]),
            sr=self.sr,
            hop_length=self.hop_length
        )

        return mel_spec_db, mel_freqs, times

    def extract_torch(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract MEL spectrogram and return as PyTorch tensor.

        Args:
            audio: Input audio signal

        Returns:
            PyTorch tensor of shape [1, mel_bins, time] (channel first for CNN)
        """
        mel_spec, _, _ = self.extract(audio)
        return torch.from_numpy(mel_spec).unsqueeze(0).float()


class SpectrogramNormalizer:
    """Normalization methods for spectrograms."""

    @staticmethod
    def min_max(data: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-8)

    @staticmethod
    def l2_norm(data: np.ndarray) -> np.ndarray:
        """L2 normalization per time frame."""
        data_normalized = SpectrogramNormalizer.min_max(data)
        return data_normalized / (np.linalg.norm(data_normalized, ord=2, axis=0, keepdims=True) + 1e-8)

    @staticmethod
    def standardize(data: np.ndarray) -> np.ndarray:
        """Z-score standardization."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-8)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Input audio signal
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
