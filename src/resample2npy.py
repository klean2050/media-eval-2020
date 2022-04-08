"""
This script takes a directory of mp3 files as input, and resamples each mp3 file
to the provided sampling rate and numpy format. To be used for downstream tasks.
"""

import os, tqdm, librosa
import argparse, numpy as np


class Processor:
    def __init__(self, fs=16000, format="mp3"):
        self.fs = fs  # Sampling rate for resampling
        self.format = format  # Typically mp3 or wav

    def iterate(self, data_path, output_path="../data/npy_audios/"):
        os.makedirs(output_path, exist_ok=True)
        self.files = [data_path + f for f in os.listdir(data_path)]
        for fn in tqdm.tqdm(self.files):
            name = fn.split("/")[-1]
            output_fn = os.path.join(output_path, name.replace(f".{self.format}", ".npy"))
            try:
                x, _ = librosa.core.load(fn, sr=self.fs)
                np.save(output_fn, x)
            except RuntimeError:
                # some audio files may be broken
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--output_path", type=str, default="../data/npy_audios")
    config = parser.parse_args()

    p = Processor(fs=16000, format="wav")
    p.iterate(config.data_path, config.output_path)
