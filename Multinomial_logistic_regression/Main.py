from Read_csv import data
from Record import Record
from Softmax import softmax

if __name__ == "__main__":
    # Converting csv to python object.
    print("We're are converting the csv table into python object...")
    records_array = []
    for line in range (0, len(data)):
        records_array.append(Record(data.Accent[line], data.Path[line]))
    softmax(records_array)