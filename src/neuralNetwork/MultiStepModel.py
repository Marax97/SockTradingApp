import tensorflow as tf
import pandas as pd
import numpy as numpy

from utils import get_files_in_directory, get_file_path
from src.dataManagement.Preprocess import preprocess_data, split_array
import src.presentation.chartDrawer as chartDrawer
import matplotlib.pyplot as plt

INDEXES_DIRECTORY = "\\resources\\stockPricesWithIndicators\\"

TRAIN_COLUMNS = ['Adj Close', 'volume_adi', 'volume_cmf', 'volume_vpt', 'trend_macd', 'trend_macd_signal',
                 'trend_macd_diff', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_vortex_ind_pos',
                 'trend_vortex_ind_neg', 'trend_vortex_diff', 'trend_cci', 'momentum_mfi', 'momentum_rsi',
                 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal']
TARGET_COLUMNS = ['Adj Close']
SPLIT_PERCENT = 85

BATCH_SIZE = 100  # liczba prób par (BUFFER_SIZE, BUFFER_SIZE)
VALIDATION_BATCH_SIZE = 30
BUFFER_SIZE = 50  # liczba dni znanych (do predykcji)
EPOCHS = 50
EVALUATION_INTERVAL = 100
DAYS_TO_PREDICT = 5


def read_indexes_prices():
    files = get_files_in_directory(INDEXES_DIRECTORY)

    index_prices = []
    for file in files:
        index = pd.read_csv(get_file_path(INDEXES_DIRECTORY + file))
        index.set_index("Date", inplace=True)
        index_prices.append(index)
    return index_prices


def generate_train_test_set(indexes, columns, start_index, future_days):
    train_set = []
    test_set = []
    for index in indexes:
        train_single, test_single = multivariate_data(index[columns], start_index, future_days)
        train_set.append(train_single)
        test_set.append(test_single)
    return train_set, test_set

def multivariate_data(dataframe, start_index, future_days):
    end_index = len(dataframe) - future_days
    return preprocess_data(dataframe[start_index:end_index], SPLIT_PERCENT)


def separate_validation_set(train_set):
    validation_set = []
    for i in range(len(train_set)):
        train_set[i], val_single = split_array(train_set[i], SPLIT_PERCENT)
        validation_set.append(val_single)
    return validation_set

def generate_random_batches(x_train_set, y_train_set, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE):
    x_batches_set = numpy.zeros(shape=(batch_size, buffer_size, len(TRAIN_COLUMNS)), dtype=numpy.float16)
    y_batches_set = numpy.zeros(shape=(batch_size, DAYS_TO_PREDICT, len(TARGET_COLUMNS)), dtype=numpy.float16)

    # x_batches_set = numpy.zeros(shape=(batch_size, buffer_size, x_train_set.shape[2]), dtype=numpy.float16)
    # y_batches_set = numpy.zeros(shape=(batch_size, DAYS_TO_PREDICT, y_train_set.shape[2]), dtype=numpy.float16)

    for i in range(batch_size):
        random_index = numpy.random.randint(len(x_train_set))
        random_pos = numpy.random.randint(len(x_train_set[random_index]) - buffer_size)
        x_batches_set[i] = x_train_set[random_index][random_pos: random_pos + buffer_size]
        y_batches_set[i] = y_train_set[random_index][
                           random_pos + buffer_size - DAYS_TO_PREDICT: random_pos + buffer_size]

    return x_batches_set, y_batches_set


def create_model(x_batches_set, y_batches_set, validation_data):
    train_data_single = tf.data.Dataset.from_tensor_slices((x_batches_set, y_batches_set))
    train_data_single = train_data_single.cache().batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices(validation_data)
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_batches_set.shape[-2:]))
    single_step_model.add(tf.keras.layers.LSTM(128, activation='relu'))
    single_step_model.add(tf.keras.layers.Dense(y_batches_set.shape[2] * DAYS_TO_PREDICT))
    single_step_model.add(tf.keras.layers.Dropout(0.01))

    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0, learning_rate=0.001), loss='mae')

    single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=5)

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
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_test_batches, y_test_batches))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    for x, y in val_data_multi.take(1):
        multi_step_plot(x[0], y[0], model.predict(x)[0])
        multi_step_plot(x[1], y[1], model.predict(x)[1])
        multi_step_plot(x[2], y[2], model.predict(x)[2])
        multi_step_plot(x[3], y[3], model.predict(x)[3])


if __name__ == "__main__":
    all_indexes = read_indexes_prices()

    x_train_set, x_test_set = generate_train_test_set(all_indexes, TRAIN_COLUMNS, 0, DAYS_TO_PREDICT)
    y_train_set, y_test_set = generate_train_test_set(all_indexes, TARGET_COLUMNS, DAYS_TO_PREDICT, 0)

    x_val_set = separate_validation_set(x_train_set)
    y_val_set = separate_validation_set(y_train_set)

    x_train_batches, y_train_batches = generate_random_batches(x_train_set, y_train_set)
    x_test_batches, y_test_batches = generate_random_batches(x_test_set, y_test_set)
    x_val_batches, y_val_batches = generate_random_batches(x_val_set, y_val_set, VALIDATION_BATCH_SIZE)

    validation_data = (x_val_batches, y_val_batches)
    model, history = create_model(x_train_batches, y_train_batches, validation_data)

    print_result()

    plot_train_history(history, "Train and validation error")
