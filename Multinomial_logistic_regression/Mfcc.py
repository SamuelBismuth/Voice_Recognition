#https://github.com/jameslyons/python_speech_features/blob/master/example.py
#https://github.com/jameslyons/python_speech_features/blob/master/README.rst

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav


def wav_to_mffc(path):  # see in details how to use it.
    (rate, sig) = wav.read(path)
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    print(fbank_feat[1:10, :])