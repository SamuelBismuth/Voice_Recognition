import Record
from Mfcc import wav_to_mffc


def from_csv_to_python_object(accent, path):
    Record.accent = accent
    Record.mfcc = wav_to_mffc(path)
    return Record


