import numpy as np

def normalize_data(data, feature_range=(0, 1)):
    min_val, max_val = feature_range
    data_min = np.min(data)
    data_max = np.max(data)
    scaled_data = (data - data_min) / (data_max - data_min)
    normalized_data = scaled_data * (max_val - min_val) + min_val
    return normalized_data, data_min, data_max

def denormalize_data(normalized_data, data_min, data_max, feature_range=(0, 1)):
    min_val, max_val = feature_range
    scaled_data = (normalized_data - min_val) / (max_val - min_val)
    data = scaled_data * (data_max - data_min) + data_min
    return data

def create_sequences(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)