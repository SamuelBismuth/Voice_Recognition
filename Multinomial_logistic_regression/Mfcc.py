#https://github.com/jameslyons/python_speech_features/blob/master/example.py
#https://github.com/jameslyons/python_speech_features/blob/master/README.rst

#from python_speech_features import delta
#from python_speech_features import logfbank
from python_speech_features import mfcc
import scipy.io.wavfile as wav


def wav_to_mfcc(path):
    """
    see in details how to use it.
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    """
    (rate, sig) = wav.read(path)
    mfcc_feat = mfcc(sig,rate)
    return mfcc_feat[1]