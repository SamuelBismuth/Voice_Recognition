from Mfcc import wav_to_mfcc


class Record:

    def __init__(self, accent, array_mfcc):
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
        self.mfcc = array_mfcc

    def to_string(self):
        print("Accent : [")
        for i in range(len(self.accent)):
            print(self.accent[i], ", ")
        print("]")
        print("Mfcc : [")
        for i in range(len(self.mfcc)):
            print(self.mfcc[i], ", ")
        print("]")