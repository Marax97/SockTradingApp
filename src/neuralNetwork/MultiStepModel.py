import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as numpy
from tensorflow.python.ops import math_ops

import utils as utils
from src.neuralNetwork.directionalMeanSquaredError import DirectionalMeanSquaredError
from src.dataManagement.preprocess import preprocess_data, split_array, remove_dataframe_rows
import src.presentation.chartDrawer as chartDrawer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

INDEXES_DIRECTORY = "\\resources\\stockPricesWithIndicators\\"
CHECKPOINT_PATH = '\\resources\\logs\\savedModel\\checkpoint.ckpt'

TRAIN_COLUMNS = ['Adj Close', 'volume_adi', 'volume_cmf', 'volume_vpt', 'trend_macd', 'trend_macd_signal',
                 'trend_macd_diff', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_vortex_ind_pos',
                 'trend_vortex_ind_neg', 'trend_vortex_diff', 'trend_cci', 'momentum_mfi', 'momentum_rsi',
                 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal']
TARGET_COLUMNS = ['Adj Close']
SPLIT_PERCENT = 85

BATCH_SIZE = 100  # liczba prób par (BUFFER_SIZE, BUFFER_SIZE) 400 100 120
VALIDATION_BATCH_SIZE = 50
TEST_BATCH_SIZE = 100
BUFFER_SIZE = 40  # liczba dni znanych (do predykcji)
EPOCHS = 20
EVALUATION_INTERVAL = BATCH_SIZE  # powinien być równy ilości próbek Batch_size
#  Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches
DAYS_TO_PREDICT = 4

CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=1e-2,
                                     patience=5,
                                     verbose=1),

    tf.keras.callbacks.TensorBoard(log_dir=utils.get_file_path('\\resources\\logs')),

    tf.keras.callbacks.ModelCheckpoint(filepath=utils.get_file_path(CHECKPOINT_PATH),
                                       save_weights_only=True,
                                       save_best_only=True,
                                       monitor='val_accuracy',
                                       mode='max',
                                       verbose=1)
]


def read_indexes_prices():
    files = utils.get_files_in_directory(INDEXES_DIRECTORY)

    index_prices = []
    for file in files:
        index = pd.read_csv(utils.get_file_path(INDEXES_DIRECTORY + file))
        index.set_index("Date", inplace=True)
        remove_dataframe_rows(index, 0, 15)
        index_prices.append(index)
    return index_prices


def generate_train_test_set(indexes, columns, start_index, future_days, data_scaler):
    train_set = []
    test_set = []
    for index in indexes:
        train_single, test_single = multivariate_data(index[columns], start_index, future_days, data_scaler)
        train_set.append(train_single)
        test_set.append(test_single)
    return train_set, test_set


def multivariate_data(dataframe, start_index, future_days, data_scaler):
    end_index = len(dataframe) - future_days
    return preprocess_data(dataframe[start_index:end_index], SPLIT_PERCENT, data_scaler)


def separate_validation_set(train_set):
    validation_set = []
    for i in range(len(train_set)):
        train_set[i], val_single = split_array(train_set[i], SPLIT_PERCENT)
        validation_set.append(val_single)
    return validation_set


def generate_random_batches(x_set, y_set, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE):
    x_batches_set = numpy.zeros(shape=(batch_size, buffer_size, len(TRAIN_COLUMNS)), dtype=numpy.float16)
    y_batches_set = numpy.zeros(shape=(batch_size, DAYS_TO_PREDICT + 1, len(TARGET_COLUMNS)), dtype=numpy.float16)

    for i in range(batch_size):
        random_index = numpy.random.randint(len(x_set))
        random_pos = numpy.random.randint(len(x_set[random_index]) - buffer_size)
        x_batches_set[i] = x_set[random_index][random_pos: random_pos + buffer_size]
        y_batches_set[i] = y_set[random_index][
                           random_pos + buffer_size - DAYS_TO_PREDICT - 1: random_pos + buffer_size]

    return x_batches_set, y_batches_set


def create_model(x_batches_set, y_batches_set, validation_data):
    train_data_tensors = tf.data.Dataset.from_tensor_slices((x_batches_set, y_batches_set))
    train_data_tensors = train_data_tensors.batch(BATCH_SIZE).repeat()
    val_data_tensors = tf.data.Dataset.from_tensor_slices(validation_data)
    val_data_tensors = val_data_tensors.batch(VALIDATION_BATCH_SIZE).repeat()

    network_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(len(TRAIN_COLUMNS),
                             return_sequences=True, input_shape=x_batches_set.shape[-2:]),
        tf.keras.layers.LSTM(2 * len(TRAIN_COLUMNS) + 1),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(y_batches_set.shape[2] * DAYS_TO_PREDICT)
    ])
    # network_model.summary()

    network_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0),
                          loss=directional_mean_squared_error,
                          # loss=DirectionalMeanSquaredError(BATCH_SIZE, DAYS_TO_PREDICT),
                          metrics=['accuracy'])

    learning_history = network_model.fit(train_data_tensors, epochs=EPOCHS,
                                         steps_per_epoch=EVALUATION_INTERVAL,
                                         validation_data=val_data_tensors,
                                         validation_steps=VALIDATION_BATCH_SIZE,
                                         callbacks=CALLBACKS)
    network_model.load_weights(utils.get_file_path(CHECKPOINT_PATH))

    return network_model, learning_history


def error_direction(y_true, y_pred):
    zeros = K.zeros([K.shape(y_true)[0], K.shape(y_true)[1]], dtype='float32')
    zeros[:, 1:] = y_true[:, :-1]
    return tf.math.multiply(K.sign(y_true - zeros), K.sign(y_pred - zeros))


def directional_mean_squared_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    # zeros = K.variable(K.zeros([K.shape(y_true)[0], K.shape(y_true)[1]], dtype='float32'))
    # zeros[:, 1:].assign(y_true[:, :-1])

    pass_days = K.variable(y_true[:, :-1], dtype='float32')
    y_true = y_true[:, 1:]

    direction = tf.math.multiply(K.sign(y_true - pass_days), K.sign(y_pred - pass_days))
    direction = tf.math.pow(K.constant(0.5), direction)
    loss = tf.math.multiply(math_ops.squared_difference(y_pred, y_true), direction)
    loss = loss[:, 1:]
    return K.mean(loss, axis=-1)


def counter_func(counter, init):
    if init:
        counter = 1.0
    else:
        counter += 1.0
    return counter


def mean_pred(y_true, y_pred):
    # loss = K.variable(K.zeros(K.shape(y_true)[0], dtype=K.dtype(y_true)))
    loss = K.variable(numpy.zeros(BATCH_SIZE, dtype='float32'))

    for i in range(BATCH_SIZE):
        wrong_direction_counter = 1.0
        for j in range(DAYS_TO_PREDICT):
            direction_true = 0
            direction_pred = 0
            if j > 0:
                direction_true = K.sign(y_true[i][j - 1] - y_true[i][j])
                direction_pred = K.sign(y_true[i][j - 1] - y_pred[i][j])

            wrong_direction_counter = tf.cond(K.not_equal(direction_true, direction_pred),
                                              lambda: counter_func(wrong_direction_counter, True),
                                              lambda: counter_func(wrong_direction_counter, False))

            loss[i].assign(K.abs(y_true[i][j] - y_pred[i][j] + (y_true[i][j] - y_pred[i][j]) * K.log(
                wrong_direction_counter)))

    return K.mean(loss)


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
    true_future = true_future[:, 1:]
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, numpy.array(history[:, 0]), label='History')
    plt.plot(numpy.arange(num_out), numpy.array(true_future), 'bo-',
             label='True Future')
    if prediction.any():
        plt.plot(numpy.arange(num_out), numpy.array(prediction), 'ro-',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def print_result(model):
    prediction = model.predict(x_test_batches)
    for i in range(6):
        x = scaler.inverse_transform(x_test_batches[i])
        y = scaler.inverse_transform(y_test_batches[i])
        z = scaler.inverse_transform(prediction[i].reshape(-1, 1))
        multi_step_plot(x, y, z)


if __name__ == "__main__":
    # Only for debugging tensors
    tf.config.experimental_run_functions_eagerly(True)
    tf.compat.v1.executing_eagerly = True

    all_indexes = read_indexes_prices()

    scaler = MinMaxScaler()
    x_train_set, x_test_set = generate_train_test_set(all_indexes, TRAIN_COLUMNS, 0, DAYS_TO_PREDICT, scaler)
    y_train_set, y_test_set = generate_train_test_set(all_indexes, TARGET_COLUMNS, DAYS_TO_PREDICT, 0,
                                                      scaler)
    x_val_set = separate_validation_set(x_train_set)
    y_val_set = separate_validation_set(y_train_set)

    x_train_batches, y_train_batches = generate_random_batches(x_train_set, y_train_set)
    x_test_batches, y_test_batches = generate_random_batches(x_test_set, y_test_set, TEST_BATCH_SIZE)
    x_val_batches, y_val_batches = generate_random_batches(x_val_set, y_val_set, VALIDATION_BATCH_SIZE)

    validation_data = (x_val_batches, y_val_batches)
    model, history = create_model(x_train_batches, y_train_batches, validation_data)

    print_result(model)
    score = model.evaluate(x_test_batches, y_test_batches, verbose=2)
    # score_dic = dict(zip(model.metrics_names, score))

    plot_train_history(history, "Train and validation error")
