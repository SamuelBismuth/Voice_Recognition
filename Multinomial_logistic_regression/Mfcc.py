#https://github.com/jameslyons/python_speech_features/blob/master/example.py
#https://github.com/jameslyons/python_speech_features/blob/master/README.rst

#from python_speech_features import delta
#from python_speech_features import logfbank
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

def wav_to_mfcc(path):
    """
    see in details how to use it.
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    """
    (rate, sig) = wav.read(path)
    mfcc_feat = mfcc(sig,rate)
    array_mfcc = []
    # Maybe find another way to get an array from the matrix.
    for i in range(len(mfcc_feat)):
        for j in range(len(mfcc_feat[0])):
            array_mfcc.append(mfcc_feat[i][j])
    # By default we want the 784 first indexes but change this.
    return array_mfcc[0:784]
