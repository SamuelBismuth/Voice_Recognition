from Mfcc import wav_to_mfcc


class Record:

    def __init__(self, accent, path):
        """
        Constructor
        :param accent: String.
        :param path: String.
        """
        if accent == "USA":
            self.accent = [1, 0, 0, 0, 0]
        if accent == "UK":
            self.accent = [0, 1, 0, 0, 0]
        if accent == "USSR":
            self.accent = [0, 0, 1, 0, 0]
        if accent == "France":
            self.accent = [0, 0, 0, 1, 0]
        if accent == "Israel":
            self.accent = [0, 0, 0, 0, 1]
        self.mfcc = wav_to_mfcc(path)

    
