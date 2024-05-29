import numpy as np


def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter='	', skip_header=1)
    return data


def preprocess_data(data):
    # needs error testing
    data = np.array(data)
    data = data[~np.isnan(data[:, 1:51].astype(float)).any(axis=1)]
    ids = data[:, 0]
    X = data[:, 1:51].astype(float)
    return ids, X
