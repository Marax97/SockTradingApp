import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import pandas as pd
import numpy as numpy
from utils import get_file_path
from src.dataManagement.Preprocess import preprocess_data
import src.presentation.chartDrawer as chartDrawer
import matplotlib.pyplot as plt

appleFile = "\\resources\\stockPricesWithIndicators\\AAPL.csv"

TARGET_COLUMNS = ['Adj Close']
SPLIT_PERCENT = 90

BATCH_SIZE = 128
BUFFER_SIZE = 256
EPOCHS = 10
EVALUATION_INTERVAL = 100

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
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

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


def plot_comparison(start_idx, length=100, train=True):
    if train:
        x = x_train
        y_true = y_train
    else:
        x = x_test
        y_true = y_test

    # End-index for the sequences.
    end_idx = start_idx + length

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = numpy.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_scaler = MinMaxScaler()
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    # For each output-signal.
    for signal in range(len(TARGET_COLUMNS)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]

        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15, 5))

        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')

        # Plot labels etc.
        plt.ylabel(TARGET_COLUMNS[signal])
        plt.legend()
        plt.show()


if __name__ == "__main__":
    appleData = readAppleDate()
    x_train, x_test = multivariate_data(appleData[['Adj Close', 'High', 'Low', 'Open', 'Close']], 0, 10)
    y_train, y_test = multivariate_data(appleData[TARGET_COLUMNS], 10, 0)

    x_val, y_val = generate_random_batches(x_train, y_train, 1)
    validation_data = (x_val, y_val)
    x_batches_set, y_batches_set = generate_random_batches(x_train, y_train)

    model, history = create_model(x_batches_set, y_batches_set, validation_data)

    result = model.evaluate(x=numpy.expand_dims(x_test, axis=0),
                            y=numpy.expand_dims(y_test, axis=0))
    plot_train_history(history, "Train and validation error")

    plot_comparison(start_idx=800, length=300, train=True)
    plot_comparison(start_idx=1150, length=100, train=False)







