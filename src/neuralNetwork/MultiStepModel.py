import tensorflow as tf
import pandas as pd
import numpy as numpy
import utils as utils
from neuralNetwork.directionalMeanSquaredError import DirectionalMeanSquaredError
import src.dataManagement.preprocess as preprocess
from pickle import dump
import matplotlib.pyplot as plt

INDEXES_DIRECTORY = "\\resources\\stockPricesWithIndicators\\"
SELECTED_COLUMNS = "\\resources\\SelectedColumns.csv"

CHECKPOINT_PATH = '\\resources\\logs\\savedModel\\checkpoint.ckpt'
ENTIRE_MODEL_PATH = '\\resources\\savedModel'
SCALER_TRAIN_PATH = '\\resources\\savedModel\\scalers\\ScalerTrain.pkl'
SCALER_TEST_PATH = '\\resources\\savedModel\\scalers\\ScalerTest.pkl'

TRAIN_COLUMNS_SIZE = 13
TARGET_COLUMNS = ['Adj Close']
SPLIT_PERCENT = 0.85

TRAIN_SAMPLES = 80000
TEST_SAMPLES = 30000
VALIDATION_SAMPLES = 26000

BATCH_SIZE = 400  # liczba prób par (BUFFER_SIZE, BUFFER_SIZE)
VALIDATION_BATCH_SIZE = 150
BUFFER_SIZE = 40  # liczba dni znanych (do predykcji)
EPOCHS = 50
DAYS_TO_PREDICT = 4

CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=1e-4,
                                     patience=5,
                                     verbose=1),

    tf.keras.callbacks.TensorBoard(log_dir=utils.get_file_path('\\resources\\logs')),

    # tf.keras.callbacks.ModelCheckpoint(filepath=utils.get_file_path(CHECKPOINT_PATH),
    #                                    save_weights_only=True,
    #                                    save_best_only=True,
    #                                    monitor='val_loss',
    #                                    mode='min',
    #                                    verbose=1)

]


def read_indexes_prices():
    files = utils.get_files_in_directory(INDEXES_DIRECTORY)

    index_prices = []
    for file in files:
        index = pd.read_csv(utils.get_file_path(INDEXES_DIRECTORY + file))
        index.set_index("Date", inplace=True)
        index = index[load_selected_features()]
        preprocess.remove_dataframe_rows(index, 0, 30)
        index_prices.append(index)
    return index_prices


def load_selected_features():
    df = pd.read_csv(utils.get_file_path(SELECTED_COLUMNS), sep='\s*,\s*', engine='python')
    return df['SelectedColumns'].astype(str).values.tolist()


def generate_train_test_set(indexes, start_index, future_days, columns=None):
    train_set = []
    test_set = []
    for index in indexes:
        if columns is not None:
            index = index[columns]
        train_single, test_single = multivariate_data(index, start_index, future_days)
        train_set.append(train_single)
        test_set.append(test_single)
    return train_set, test_set


def multivariate_data(dataframe, start_index, future_days):
    end_index = len(dataframe) - future_days
    return preprocess.split_data(dataframe[start_index:end_index], SPLIT_PERCENT)


def separate_validation_set(train_set):
    validation_set = []
    for i in range(len(train_set)):
        train_set[i], val_single = preprocess.split_array(train_set[i], SPLIT_PERCENT)
        validation_set.append(val_single)
    return validation_set


def generate_random_samples(x_set, y_set, number_of_samples=TRAIN_SAMPLES, buffer_size=BUFFER_SIZE):
    x_batches_set = numpy.zeros(shape=(number_of_samples, BUFFER_SIZE, TRAIN_COLUMNS_SIZE), dtype=numpy.float32)
    y_batches_set = numpy.zeros(shape=(number_of_samples, DAYS_TO_PREDICT + 6, len(TARGET_COLUMNS)),
                                dtype=numpy.float32)

    for i in range(number_of_samples):
        random_index = numpy.random.randint(len(x_set))
        random_pos = numpy.random.randint(len(x_set[random_index]) - buffer_size)
        x_batches_set[i] = x_set[random_index][random_pos: random_pos + buffer_size]
        y_batches_set[i] = y_set[random_index][
                           random_pos + buffer_size - DAYS_TO_PREDICT - 6: random_pos + buffer_size]

    return x_batches_set, y_batches_set


def create_model(x_batches_set, y_batches_set, validation_data):
    network_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(TRAIN_COLUMNS_SIZE, return_sequences=True, input_shape=(BUFFER_SIZE, TRAIN_COLUMNS_SIZE)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(TRAIN_COLUMNS_SIZE * 2, return_sequences=True),
        tf.keras.layers.LSTM(2 * DAYS_TO_PREDICT + 1),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(y_batches_set.shape[2] * DAYS_TO_PREDICT),
        tf.keras.layers.Reshape([y_batches_set.shape[2] * DAYS_TO_PREDICT, len(TARGET_COLUMNS)])
    ])

    network_model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=DirectionalMeanSquaredError())

    learning_history = network_model.fit(x=x_batches_set, y=y_batches_set, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                         validation_data=validation_data, validation_batch_size=VALIDATION_BATCH_SIZE,
                                         steps_per_epoch=BATCH_SIZE, validation_steps=VALIDATION_BATCH_SIZE,
                                         callbacks=CALLBACKS)

    # network_model.load_weights(utils.get_file_path(CHECKPOINT_PATH))

    return network_model, learning_history


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


def save_model():
    score = model.evaluate(x_test_batches, y_test_batches, verbose=2)
    savedScore = None
    if len(utils.get_files_in_directory(ENTIRE_MODEL_PATH)):
        savedModel = tf.keras.models.load_model(utils.get_file_path(ENTIRE_MODEL_PATH),
                                                custom_objects={'loss': DirectionalMeanSquaredError()}, compile=False)
        savedModel.compile(optimizer=tf.keras.optimizers.Adam(), loss=DirectionalMeanSquaredError())
        savedScore = savedModel.evaluate(x_test_batches, y_test_batches, verbose=0)
        if savedScore < score:
            print("Model did not improve on test set. Best score was {} but model get {}".format(savedScore, score))
            return
    model.save(utils.get_file_path(ENTIRE_MODEL_PATH))
    dump(train_scaler, open(utils.get_file_path(SCALER_TRAIN_PATH), 'wb'))
    dump(test_scaler, open(utils.get_file_path(SCALER_TEST_PATH), 'wb'))
    print("New model was saved due to improve on test set from {} to {}".format(savedScore, score))


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    all_indexes = read_indexes_prices()

    x_train_set, x_test_set = generate_train_test_set(all_indexes, 0, DAYS_TO_PREDICT)
    y_train_set, y_test_set = generate_train_test_set(all_indexes, DAYS_TO_PREDICT, 0, TARGET_COLUMNS)
    x_val_set = separate_validation_set(x_train_set)
    y_val_set = separate_validation_set(y_train_set)

    train_scaler = preprocess.config_scaler(x_train_set)
    test_scaler = preprocess.config_scaler(y_train_set)
    x_train_set, x_val_set, x_test_set = preprocess.normalize_list_of_data([x_train_set, x_val_set, x_test_set],
                                                                           train_scaler)
    y_train_set, y_val_set, y_test_set = preprocess.normalize_list_of_data([y_train_set, y_val_set, y_test_set],
                                                                           test_scaler)

    x_train_batches, y_train_batches = generate_random_samples(x_train_set, y_train_set)
    x_test_batches, y_test_batches = generate_random_samples(x_test_set, y_test_set, TRAIN_SAMPLES)
    x_val_batches, y_val_batches = generate_random_samples(x_val_set, y_val_set, VALIDATION_SAMPLES)

    validation_data = (x_val_batches, y_val_batches)
    model, history = create_model(x_train_batches, y_train_batches, validation_data)

    plot_train_history(history, "Train and validation error")
    print(model.summary())
    save_model()
