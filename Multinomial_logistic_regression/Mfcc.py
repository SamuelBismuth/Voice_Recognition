from python_speech_features import mfcc
import scipy.io.wavfile as wav


def wav_to_mfcc(path):
    (rate, sig) = wav.read(path)
    mfcc_feat = mfcc(sig,rate)
    array_mfcc = []
    # Maybe the data is dafuq by the conversion from matrix to array.
    # Maybe find another way to get an array from the matrix.
    for i in range(len(mfcc_feat)):
        for j in range(len(mfcc_feat[0])):
            array_mfcc.append(mfcc_feat[i][j])
    # print(len(array_mfcc))
    return array_mfcc
