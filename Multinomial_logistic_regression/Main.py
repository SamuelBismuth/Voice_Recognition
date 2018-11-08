from Read_csv import data
from Run_csv_to_object import from_csv_to_python_object

if __name__ == "__main__":
    # Converting csv to python object.
    print("We're are converting the csv table into python object...")
    for line in data:
        from_csv_to_python_object(data.Accent[1], data.Pathx[1])
