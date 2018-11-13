from random import shuffle

from Mfcc import wav_to_mfcc
from Read_csv import data
from Record import Record
from Softmax import softmax
from prepare_wav import song_time, divide_audio

if __name__ == "__main__":
    # Converting csv to python object.
    print("We're are converting the csv table into python object...")
    records_array = []
    features = 100
    for line in range(0, len(data)):
        for i in range(0, int(song_time(data.Path[line])), 5):
            records_array.append(Record(data.Accent[line], wav_to_mfcc(divide_audio(data.Path[line], i, i + 5))))
    print(len(records_array))
    # for i in range(len(records_array)):
        # records_array[i].to_string()
    shuffle(records_array)
    softmax(records_array)
