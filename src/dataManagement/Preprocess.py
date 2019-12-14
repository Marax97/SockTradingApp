from sklearn.preprocessing import MinMaxScaler


def split_array(array, percentage):
    splitIndex = int(len(array) * percentage / 100);
    return array[:splitIndex], array[splitIndex:]


def split_data(dataFrame, percentage):
    return split_array(dataFrame.values, percentage)


def normalize_data(train_data, test_data):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled


def preprocess_data(dataFrame, split_percent):
    train, test = split_data(dataFrame, split_percent)
    return normalize_data(train, test)
