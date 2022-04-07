"""
This script takes a directory of mp3 files as input, and resamples each mp3 file
to the provided sampling rate and numpy format. To be used for downstream tasks.
"""

import os, fire, tqdm
import torchaudio, torch
import numpy as np, torch.nn as nn


class ChangeSampleRate(nn.Module):
    """
    https://discuss.pytorch.org/t/change-sample-rate-using-torchaudio/23177/7
    """

    def __init__(self, input_rate: int, output_rate: int):
        super().__init__()
        self.output_rate = output_rate
        self.input_rate = input_rate

    def forward(self, wav: torch.tensor) -> torch.tensor:
        # Only accepts 1-channel waveform input
        wav = wav.view(wav.size(0), -1)
        new_length = wav.size(-1) * self.output_rate // self.input_rate
        indices = torch.arange(new_length) * (self.input_rate / self.output_rate)
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        return round_down * (1.0 - indices.fmod(1.0)).unsqueeze(
            0
        ) + round_up * indices.fmod(1.0).unsqueeze(0)


class Processor:
    def __init__(self, fs=16000, format="mp3"):
        self.fs = fs  # Sampling rate for resampling
        self.format = format  # Typically mp3 or wav

    def get_paths(self, data_path, output_path):
        # Find all files contained in the given directory
        self.files = [data_path + f for f in os.listdir(data_path)]
        # os.path.join(data_path, f"**.{format}"))#recursive=True)
        self.npy_path = os.path.join(output_path, "npy")
        os.makedirs(output_path, exist_ok=True)

    def get_npy(self, fn):
        # Resample a file
        x, sr = torchaudio.load(fn)
        x = x.sum(axis=0) / 2
        csr = ChangeSampleRate(sr, self.fs)
        out_wav = csr(x.unsqueeze(0))
        return out_wav.numpy()

    def iterate(self, data_path):
        output_path = "../data/npy_audios/"  # Replace with desired output directory
        self.get_paths(data_path, output_path)
        for fn in tqdm.tqdm(self.files):
            output_fn = os.path.join(output_path, fn.replace(f".{format}", ".npy"))
            try:
                x = self.get_npy(fn)
                np.save(open(output_fn, "wb"), x)
            except RuntimeError:
                # some audio files are broken
                continue


if __name__ == "__main__":

    p = Processor(fs=16000, format="wav")
    fire.Fire({"run": p.iterate})
