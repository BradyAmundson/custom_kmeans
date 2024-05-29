import numpy as np


def load_data(file_path):
    """
    Load data from a file.

    Parameters:
    - file_path (str): Path to the file containing the data.

    Returns:
    - data (np.ndarray): Loaded data as a NumPy array.
    """
    data = np.genfromtxt(file_path, delimiter='	', skip_header=1)
    return data


def preprocess_data(data):
    """
    Preprocess the data by removing rows with NaN values.

    Parameters:
    - data (np.ndarray): Input data array.

    Returns:
    - ids (np.ndarray): Array of IDs (first column of the data).
    - X (np.ndarray): Preprocessed data array with only valid rows and selected columns.
    """
    data = np.array(data)
    data = data[~np.isnan(data[:, 1:51].astype(float)).any(axis=1)]
    ids = data[:, 0]
    X = data[:, 1:51].astype(float)
    return ids, X
