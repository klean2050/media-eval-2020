'''
This script takes a directory of mp3 files as input, and resamples each mp3 file
to the provided sampling rate and numpy format. To be used for downstream tasks.
'''

import os, glob, fire, tqdm
import numpy as np, librosa

class Processor:
	def __init__(self, fs=16000, format="mp3"):
		self.fs = fs # Sampling rate for resampling
		self.format = format # Typically mp3 or wav

    # Find all files contained in the given directory
	def get_paths(self, data_path, output_path):
		self.files = glob.glob(os.path.join(data_path, f'**/*.{format}'), recursive=True)
		self.npy_path = os.path.join(output_path, 'npy')
		os.makedirs(output_path, exist_ok=True)
    
    # Resample a file
	def get_npy(self, fn):
		x, _ = librosa.core.load(fn, sr=self.fs)
		return x

	def iterate(self, data_path):
		output_path = './npy_audios' # Replace with desired output directory
		self.get_paths(data_path, output_path)
		for fn in tqdm.tqdm(self.files):
			output_fn = os.path.join(output_path, fn.replace(f".{format}", ".npy"))
			if not os.path.exists(output_fn):
				try:
					x = self.get_npy(fn)
					np.save(open(output_fn, 'wb'), x) # Save in npy format
				except RuntimeError:
					# some audio files are broken
					print(fn)
					continue

if __name__ == '__main__':

	p = Processor(fs=16000, format="wav")
	fire.Fire({'run': p.iterate})
