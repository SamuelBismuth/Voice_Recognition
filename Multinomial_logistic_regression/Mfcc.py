from python_speech_features import mfcc
import scipy.io.wavfile as wav


def wav_to_mfcc(path):
    (rate, sig) = wav.read(path)
    mfcc_feat = mfcc(sig,rate)
    return mfcc_feat
