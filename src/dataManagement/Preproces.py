
from sklearn import preprocessing

def splitToTrainAndTest(dataFrame, percentage):
    splitindex = int(len(dataFrame) * percentage);
    return dataFrame[:splitindex], dataFrame[splitindex:]

def normalizeData(dataFrame):
    result = dataFrame.copy()
    for columnName in dataFrame.columns:
        min_max_scaler = preprocessing.MinMaxScaler()
        result[columnName] = min_max_scaler.fit_transform(dataFrame[columnName])
    return result
