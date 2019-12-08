
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as numpy
from utils import get_file_path
from src.dataManagement.Preprocess import preprocess_data
import src.presentation.chartDrawer as chartDrawer
import matplotlib.pyplot as plt

appleFile = "\\resources\\stockPricesWithIndicators\\AAPL.csv"

target_columns = ["Adj Close"]
days_to_predict = 10

BATCH_SIZE = 256
BUFFER_SIZE = 10000

def readAppleDate():
    appleData = pd.read_csv(get_file_path(appleFile))
    appleData = preprocess.normalizeData(appleData)
    return preprocess.splitToTrainAndTest(appleData, 80)

def univariate_data(dataset, start_index, end_index, target_size):
    return dataset[start_index:end_index], dataset[end_index:end_index+target_size]

def train_model(currentData, futureDate):
    currentData = currentData["Adj Close"]
    futureDate = futureDate["Adj Close"]
    pd.Series()
    predicted = pd.Series(baseline(currentData),index=futureDate.index)

    train_univariate = tf.data.Dataset.from_tensor_slices((currentData, futureDate))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((currentData, futureDate))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=currentData.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')

    for x, y in val_univariate.take(1):
        print(simple_lstm_model.predict(x).shape)

    plot_univariate(currentData, futureDate, predicted)

def plot_univariate(historical, future, predicted):
    plt.plot(historical)
    plt.plot(future, marker='o')
    plt.plot(predicted, marker='x')
    print(future)
    print(predicted)
    plt.show()

def baseline(history):
  return history.mean()

def plot_model(currentData, futureDate):
    chartDrawer.plot_date_prices(currentData['Adj Close'], futureDate['Adj Close'], "Simple Example")

def neural_network(train, test):
    train_univariate = tf.data.Dataset.from_tensor_slices((train, test))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()


if __name__ == "__main__":
    train, test = readAppleDate()
    #plot_model(train, test)
    history, future = univariate_data(train, 20, 100, 1)
    train_model(history, future)

# data = pd.read_csv(get_file_path(appleFile))
# plt.plot(data.Close)
#  plt.plot(data.trend_macd, label='Macd')
#  plt.plot(data.trend_macd_signal, label='Macd signal')
#  #plt.bar(data.trend_macd_diff, label='Macd difference')
#  data.trend_macd_diff.plot.bar(label='Macd difference')
#  plt.title('Moving Average Convergence Divergence')
#  plt.legend()
#  plt.show()
