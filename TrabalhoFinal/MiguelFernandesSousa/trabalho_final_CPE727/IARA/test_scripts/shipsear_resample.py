import os
import numpy as np
import tqdm
import scipy.io.wavfile as scipy_wav

import iara.processing.analysis as iara_analysis


directory_in = './data/shipsear'
directory_out = './data/shipsear_16e3'
ref_freq = int(16e3)

os.makedirs(directory_out, exist_ok=True)

wav_files = [f for f in os.listdir(directory_in) if f.endswith('.wav')]

for wav_file in tqdm.tqdm(wav_files, desc='Files', leave=False):
    file_in = os.path.join(directory_in, wav_file)
    file_out = os.path.join(directory_out, wav_file)

    fs, data = scipy_wav.read(file_in)

    data = iara_analysis.decimate(data, fs/ref_freq)
    data = data.astype(np.int32)

    scipy_wav.write(file_out, ref_freq, data)
