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
BUFFER_SIZE = 80
EPOCHS = 5
EVALUATION_INTERVAL = 80
STEP = 1

def readAppleDate():
    appleData = pd.read_csv(get_file_path(appleFile))
    appleData.set_index("Date", inplace=True)
    return appleData

def multivariate_data(dataframe, start_index, future_days):
    end_index = len(dataframe) - future_days
    return preprocess_data(dataframe[start_index:end_index], SPLIT_PERCENT)

def generate_random_batches(x_train_set, y_train_set, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE):
    x_batches_set = numpy.zeros(shape=(batch_size, buffer_size, x_train_set.shape[1]), dtype=numpy.float16)
    y_batches_set = numpy.zeros(shape=(batch_size, buffer_size, y_train_set.shape[1]), dtype=numpy.float16)

    for i in range(batch_size):
        random_index = numpy.random.randint(x_train_set.shape[0] - buffer_size)
        x_batches_set[i] = x_train_set[random_index:random_index+buffer_size]
        y_batches_set[i] = y_train_set[random_index:random_index + buffer_size]

    return x_batches_set, y_batches_set

def create_model(x_batches_set, y_batches_set,validation_data):
    train_data_single = tf.data.Dataset.from_tensor_slices((x_batches_set, y_batches_set))
    train_data_single = train_data_single.cache().batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices(validation_data)
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_batches_set.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(y_batches_set.shape[2]))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

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


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, x, marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, x.flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    plt.show()
    return plt


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, numpy.array(history[:, 1]), label='History')
    plt.plot(numpy.arange(num_out) / STEP, numpy.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(numpy.arange(num_out) / STEP, numpy.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def print_result():
    val_data_single = tf.data.Dataset.from_tensor_slices(validation_data)
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    for x, y in val_data_single.take(1):
        show_plot([x[0][:, 0].numpy(), y[0][-1].numpy(),
                   model.predict(x)[0]], 10,
                  'Single Step Prediction')
        show_plot([x[1][:, 0].numpy(), y[1][-1].numpy(),
                   model.predict(x)[1]], 10,
                  'Single Step Prediction')
        show_plot([x[2][:, 0].numpy(), y[2][-1].numpy(),
                   model.predict(x)[2]], 10,
                  'Single Step Prediction')
        show_plot([x[3][:, 0].numpy(), y[3][-1].numpy(),
                   model.predict(x)[3]], 10,
                  'Single Step Prediction')


if __name__ == "__main__":
    appleData = readAppleDate()

    x_train, x_test = multivariate_data(appleData[['Adj Close', 'High', 'Low', 'Open', 'Close']], 0, 10)
    y_train, y_test = multivariate_data(appleData[TARGET_COLUMNS], 10, 0)

    x_train, x_val = split_array(x_train, SPLIT_PERCENT)
    y_train, y_val = split_array(y_train, SPLIT_PERCENT)

    x_batches_set, y_batches_set = generate_random_batches(x_train, y_train)
    x_val_set, y_val_set = generate_random_batches(x_val, y_val, 10)
    validation_data = (x_val_set, y_val_set)

    model, history = create_model(x_batches_set, y_batches_set, validation_data)

    print_result()

    plot_train_history(history, "Train and validation error")

    # show_plot([x_train[1000:, 0], y_train[-1, 0], model.predict()], 0,
    #           'Baseline Prediction Example')

    # for x, y in train_data_multi.take(1):
    #     multi_step_plot(x[0], y[0], np.array([0]))
