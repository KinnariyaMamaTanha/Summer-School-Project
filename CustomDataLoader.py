from torch.utils.data import Dataset

from PreprocessingUtils import AudioUtil


# Sound dataset
class SoundDS(Dataset):
    """SoundDS."""

    def __init__(self, df, data_path) -> None:
        """__init__.

        :param self:
        :param df:
        :type df: DataFrame
        :param data_path:
        :type data_path: Path
        :rtype: None
        """
        super().__init__()
        self.df = df
        self.data_path = str(data_path)
        self.duration: int = 4000
        self.sr: int = 44100
        self.channel: int = 2
        self.shift_pct: float = 0.4

    def __len__(self):
        """Number of items in dataset.

        :param self:
        """
        return len(self.df)

    def __getitem__(self, index):
        """Get i'th item in dataset.

        :param self:
        :param index:
        """
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + self.df.loc[index, "relative_path"]
        # Get the classID
        class_id = self.df.loc[index, "classID"]

        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(
            sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
        )
        return (aug_sgram, class_id)
