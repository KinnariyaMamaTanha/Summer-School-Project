from typing import Tuple
import random
import torch
import torchaudio
from torchaudio import transforms


class AudioUtil:
    """AudioUtil."""

    @staticmethod
    def open(audio_file):
        """Load an audio file. Return the signal as a tensor and the sample rate.

        :param audio_file:
        """
        sig: torch.Tensor
        sr: int
        sig, sr = torchaudio.load(audio_file)  # sig: signal, sr: sample rate
        return (sig, sr)

    @staticmethod
    def rechannel(aud, new_channel):
        """Convert the given audio to the desired number of channels.

        :param aud:
        :param new_channel:
        """
        sig, sr = aud
        if sig.shape[0] == new_channel:
            # do not need do anything
            return aud

        if new_channel == 1:
            resig = sig[:1, :]  # Only reserve the first channel
        else:
            resig = torch.cat([sig, sig])  # extend 1 channel to 2 channels

        return (resig, sr)

    @staticmethod
    def resample(aud: Tuple[torch.Tensor, int], new_sr: int):
        """Since Resample applies to a single channel, we resample one channel at a time.

        :param aud:
        :param new_sr:
        """
        sig, sr = aud
        if sr == new_sr:
            return aud

        resig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])
        num_channels = sig.shape[0]
        if num_channels > 1:
            # resample the second channel only when it exists
            retwo = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])
            resig = torch.cat([resig, retwo])

        return (resig, new_sr)

    @staticmethod
    def pad_trunc(aud: Tuple[torch.Tensor, int], max_ms: int):
        """Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds.

        :param aud:
        :param max_ms:
        """
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            sig: torch.Tensor = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len: int = random.randint(0, max_len - sig_len)
            pad_end_len: int = max_len - sig_len - pad_begin_len

            pad_begin: torch.Tensor = torch.zeros((num_rows, pad_begin_len))
            pad_end: torch.Tensor = torch.zeros((num_rows, pad_end_len))

            sig: torch.Tensor = torch.cat((pad_begin, sig, pad_end), dim=1)

        return (sig, sr)

    @staticmethod
    def time_shift(aud: Tuple[torch.Tensor, int], shift_limit: float):
        """Shifts the signal to the left or right by some percent.
           Values at the end are 'wrapped around' to the start of the transformed signal.

        :param aud:
        :type aud: Tuple[torch.Tensor, int]
        :param shift_limit:
        """
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectro_gram(
        aud: Tuple[torch.Tensor, int], n_mels: int = 64, n_fft: int = 1024, hop_len=None
    ):
        """Generate a Spectrogram.

        :param aud:
        :type aud: Tuple[torch.Tensor, int]
        :param n_mels:
        :type n_mels: int
        :param n_fft:
        :type n_fft: int
        :param hop_len:
        """
        sig, sr = aud

        spec = transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
        )(sig)

        top_db = 80

        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spectro_augment(
        spec, max_mask_pct: float = 0.1, n_freq_masks: int = 1, n_time_masks: int = 1
    ):
        """spectro_augment.

        :param spec:
        :param max_mask_pct:
        :type max_mask_pct: float
        :param n_freq_masks:
        :type n_freq_masks: int
        :param n_time_masks:
        :type n_time_masks: int
        """
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()

        aug_spec = spec
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(
                aug_spec, mask_value
            )

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
