from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def remove_dataframe_rows(dataframe, from_index, to_index):
    dataframe.drop(dataframe.index[from_index:to_index])


def split_array(array, percentage):
    splitIndex = int(len(array) * percentage);
    return array[:splitIndex], array[splitIndex:]


def split_data(dataFrame, percentage):
    return split_array(dataFrame.values, percentage)


def config_scaler(train_data):
    allMerged = np.concatenate(train_data)
    scaler = StandardScaler()
    scaler.fit_transform(allMerged)
    return scaler


def scale_data(data, scaler):
    data = scaler.transform(data)
    return data


def normalize_list_of_data(list_of_sets, scaler):
    for i in range(len(list_of_sets)):
        for j in range(len(list_of_sets[i])):
            list_of_sets[i][j] = scale_data(list_of_sets[i][j], scaler)
    return list_of_sets
