import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as numpy
from utils import get_file_path
from src.dataManagement.Preprocess import preprocess_data, split_array
import src.presentation.chartDrawer as chartDrawer
import matplotlib.pyplot as plt

appleFile = "\\resources\\stockPricesWithIndicators\\AAPL.csv"

TARGET_COLUMNS = ['Adj Close']
SPLIT_PERCENT = 85

BATCH_SIZE = 64
BUFFER_SIZE = 100
EPOCHS = 8
EVALUATION_INTERVAL = 80
DAYS_TO_PREDICT = 10


def readAppleDate():
    appleData = pd.read_csv(get_file_path(appleFile))
    appleData.set_index("Date", inplace=True)
    return appleData


def multivariate_data(dataframe, start_index, future_days):
    end_index = len(dataframe) - future_days
    return preprocess_data(dataframe[start_index:end_index], SPLIT_PERCENT)


def generate_random_batches(x_train_set, y_train_set, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE):
    x_batches_set = numpy.zeros(shape=(batch_size, buffer_size, x_train_set.shape[1]), dtype=numpy.float16)
    y_batches_set = numpy.zeros(shape=(batch_size, DAYS_TO_PREDICT, y_train_set.shape[1]), dtype=numpy.float16)

    for i in range(batch_size):
        random_index = numpy.random.randint(x_train_set.shape[0] - buffer_size)
        x_batches_set[i] = x_train_set[random_index:random_index + buffer_size]
        y_batches_set[i] = y_train_set[random_index + buffer_size - DAYS_TO_PREDICT: random_index + buffer_size]

    return x_batches_set, y_batches_set


def create_model(x_batches_set, y_batches_set, validation_data):
    train_data_single = tf.data.Dataset.from_tensor_slices((x_batches_set, y_batches_set))
    train_data_single = train_data_single.cache().batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices(validation_data)
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_batches_set.shape[-2:]))
    single_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    single_step_model.add(tf.keras.layers.Dense(y_batches_set.shape[2] * DAYS_TO_PREDICT))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=50)

    return single_step_model, single_step_history


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, numpy.array(history[:, 1]), label='History')
    plt.plot(numpy.arange(num_out), numpy.array(true_future), 'bo-',
             label='True Future')
    if prediction.any():
        plt.plot(numpy.arange(num_out), numpy.array(prediction), 'ro-',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def print_result():
    val_data_multi = tf.data.Dataset.from_tensor_slices(validation_data)
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    for x, y in val_data_multi.take(1):
        multi_step_plot(x[0], y[0], model.predict(x)[0])
        multi_step_plot(x[1], y[1], model.predict(x)[1])
        multi_step_plot(x[2], y[2], model.predict(x)[2])
        multi_step_plot(x[3], y[3], model.predict(x)[3])


if __name__ == "__main__":
    appleData = readAppleDate()

    x_train, x_test = multivariate_data(appleData[['Adj Close', 'High', 'Low', 'Open', 'Close']], 0, DAYS_TO_PREDICT)
    y_train, y_test = multivariate_data(appleData[TARGET_COLUMNS], DAYS_TO_PREDICT, 0)

    x_train, x_val = split_array(x_train, SPLIT_PERCENT)
    y_train, y_val = split_array(y_train, SPLIT_PERCENT)

    x_batches_set, y_batches_set = generate_random_batches(x_train, y_train)
    x_val_set, y_val_set = generate_random_batches(x_val, y_val, 10)
    validation_data = (x_val_set, y_val_set)

    model, history = create_model(x_batches_set, y_batches_set, validation_data)

    print_result()

    plot_train_history(history, "Train and validation error")
