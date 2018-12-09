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
    features = 3874
    seconds = 3

    for line in range(0, len(data)):
        song_final_time = int(song_time(data.Path[line])) - int(song_time(data.Path[line])) % seconds
        for i in range(0, song_final_time, seconds):
            records_array.append(Record(data.Accent[line], wav_to_mfcc(divide_audio(data.Path[line], i, i + seconds))[0:features]))
            
    start_passed_info = 0
    end_passed_info = 0
    chunk=500
    i=0
    print(len(records_array))
    while(end_passed_info<len(records_array)):
        if(end_passed_info+chunk<=len(records_array)):
            end_passed_info=end_passed_info+chunk
        else:
            end_passed_info = len(records_array)
        with open('Data\data'+str(i)+'.txt', 'wb') as fp:
            pickle.dump(records_array[start_passed_info:end_passed_info], fp)
        start_passed_info = end_passed_info
        i+=1
    j=0
    records_array =[]
    while(j<11):
        with open ('Data\data'+str(j)+'.txt', 'rb') as fp:
            records_array.extend(pickle.load(fp))
        j+=1 
    print(len(records_array))
    print ("Done")
    shuffle(records_array)
    softmax(records_array)
