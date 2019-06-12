import numpy as np
import librosa
import io
import os
import matplotlib.pyplot as plt
import uuid

from pydub import AudioSegment
from scipy.signal import stft
from six.moves.urllib.request import urlopen

# sample rate
SR = 16000
URL = os.environ['FLASK_APP_URL']

def plotSpectrogram(wav):
    name = 'static/' + uuid.uuid4().hex + '.jpg'
    plt.specgram(wav, Fs=SR)
    plt.savefig(os.path.join(os.path.dirname(__file__), name))

    return URL + '/' + name

def read_wav_file(x):
    data = AudioSegment.from_file((io.BytesIO(urlopen(x).read())))
    samplerate = data.frame_rate

    data = data.get_array_of_samples()
    data = np.array(data).astype('float32')

    if samplerate > SR:
        data = librosa.resample(data, samplerate, SR)

    data = data / np.iinfo(np.int16).max

    # return wav
    return data

# source with some changes https://github.com/dawidkopczyk/speech_recognition/blob/master/dataset.py
def process_wav_file(x, threshold_freq=5500, eps=1e-10):
    # Read wav file to array
    wav = read_wav_file(x)
    # Sample rate
    L = SR

    # If longer then randomly truncate
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i+L)]  

    # If shorter then randomly add silence
    elif len(wav) < L:
        rem_len = L - len(wav)
        silence_part = np.random.randint(-100,100,16000).astype(np.float32) / np.iinfo(np.int16).max
        j = np.random.randint(0, rem_len)
        silence_part_left  = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])
    
    # Create spectrogram using discrete FFT (change basis to frequencies)
    freqs, times, spec = stft(wav, L, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)

    # save sepctograma
    img = plotSpectrogram(wav)

    # Cut high frequencies
    if threshold_freq is not None:
        spec = spec[freqs <= threshold_freq,:]
        freqs = freqs[freqs <= threshold_freq]
    # Log spectrogram
    amp = np.log(np.abs(spec)+eps)
    
    return np.expand_dims(amp, axis=2), img