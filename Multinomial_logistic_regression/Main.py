from random import shuffle

from Mfcc import wav_to_mfcc
from Read_csv import data
from Record import Record
from Softmax import softmax
from prepare_wav import song_time, divide_audio
import pickle

if __name__ == "__main__":
    # Converting csv to python object.
    print("We're converting the csv table into python object...")
    records_array = []
    features = 6450
    seconds = 5

    for line in range(0, len(data)):
        song_final_time = int(song_time(data.Path[line])) - int(song_time(data.Path[line])) % seconds
        for i in range(0, song_final_time, seconds):
            records_array.append(Record(data.Accent[line], wav_to_mfcc(divide_audio(data.Path[line], i, i + seconds))[0:features]))
    with open('data.txt', 'wb') as fp:
        pickle.dump(records_array, fp)
    with open ('data.txt', 'rb') as fp:
        records_array = pickle.load(fp)
    print ("Done")
    shuffle(records_array)
    softmax(records_array)
