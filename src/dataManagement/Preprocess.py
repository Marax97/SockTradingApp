from sklearn.preprocessing import MinMaxScaler



def splitToTrainAndTest(dataFrame, percentage):
    splitIndex = int(len(dataFrame) * percentage/100);
    return dataFrame.values[:splitIndex], dataFrame.values[splitIndex:]

def normalizeData(train_data, test_data):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return  train_scaled, test_scaled


def preprocess_data(dataFrame, split_percent):
    train, test = splitToTrainAndTest(dataFrame, split_percent)
    return normalizeData(train, test)