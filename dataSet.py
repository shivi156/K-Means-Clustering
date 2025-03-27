import numpy as np
import csv

class DataSet:
    def __init__(self, filename: str, column_index: list[int]):
        self.filename = filename
        self.column_index = column_index

    def normalise_data(self, x):
        min_values = x.min(axis=0)
        max_values = x.max(axis=0)
        return (x - min_values) / (max_values - min_values)

    def load_data(self):
        data = []
        with open(self.filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                data.append([float(row[i]) for i in self.column_index])
        data = np.array(data)
        return self.normalise_data(data)
