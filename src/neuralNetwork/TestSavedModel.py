import tensorflow as tf
import sys
import matplotlib
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops

from neuralNetwork.directionalMeanSquaredError import DirectionalMeanSquaredError

matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from datetime import datetime

import numpy as numpy
import utils as utils
import pandas as pd
from utils import get_file_path

ENTIRE_MODEL_PATH = "\\resources\\logs\\savedModel\\entire\\model"
INDEXES_DIRECTORY = "\\resources\\stockPricesWithIndicators\\"
SELECTED_COLUMNS = "\\resources\\SelectedColumns.csv"


def get_symbols_from_csv():
    df = pd.read_csv(get_file_path(SELECTED_COLUMNS), sep='\s*,\s*', engine='python')
    return df['SelectedColumns'].astype(str).values.tolist()


def read_indexes_prices():
    files = utils.get_files_in_directory(INDEXES_DIRECTORY)

    index_prices = {}
    for file in files:
        index = pd.read_csv(utils.get_file_path(INDEXES_DIRECTORY + file))
        index.set_index("Date", inplace=True)
        index = index[get_symbols_from_csv()]
        index_prices[file[:file.rfind(".")]] = index
    return index_prices


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, indexes, model):
        super().__init__()
        self.indexes = indexes
        self.model = model

        sc = MplCanvas(self, width=5, height=4, dpi=100)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.init_select_index(indexes))
        layout.addWidget(sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowTitle('Stock trading app')
        self.show()

    def on_predict(self):
        selected_day = self.date_calendar.date().toString("yyyy-MM-dd")
        selected_index = self.indexes.get(self.index_combo.currentText())
        datetime.strptime(selected_day, "%Y-%m-%d")
        date_index = selected_index.loc[selected_day]
        past_days = selected_index[:selected_day].iloc[-40:]
        future_days = selected_index[selected_day:].iloc[:4]

        prediction = self.model.predict(numpy.array([past_days.to_numpy()]))[0]
        x = 5

    def init_select_index(self, indexes):
        gridLayout = QtWidgets.QGridLayout()

        self.index_label = QtWidgets.QLabel(self)
        self.index_label.setText("Select index: ")
        gridLayout.addWidget(self.index_label, 1, 1)
        self.index_combo = QtWidgets.QComboBox(self)
        self.index_combo.addItems(indexes.keys())
        gridLayout.addWidget(self.index_combo, 1, 2)

        self.date_label = QtWidgets.QLabel(self)
        self.date_label.setText("Select date: ")
        gridLayout.addWidget(self.date_label, 2, 1)
        self.date_calendar = QtWidgets.QDateEdit(calendarPopup=True)
        # self.date_calendar.setDate(QtCore.QDateTime.currentDateTime())
        start_date = datetime.strptime(indexes[next(iter(indexes.keys()))].index[60], "%Y-%m-%d")
        end_date = datetime.strptime(indexes[next(iter(indexes.keys()))].index[-5], "%Y-%m-%d")

        self.date_calendar.setMinimumDate(QtCore.QDate(start_date))
        self.date_calendar.setMaximumDate(QtCore.QDate(end_date))
        gridLayout.addWidget(self.date_calendar, 2, 2)

        self.predict_button = QtWidgets.QPushButton("Predict")
        self.predict_button.clicked.connect(lambda: self.on_predict())
        # QtCore.QObject.connect(self.predict_button, QtCore.SIGNAL('clicked()'), self.on_predict())
        gridLayout.addWidget(self.predict_button, 3, 1, 1, 2)

        grid_widget = QtWidgets.QWidget()
        grid_widget.setLayout(gridLayout)
        grid_widget.setMaximumWidth(300)
        grid_widget.setMaximumHeight(100)
        return grid_widget


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
    return K.mean(loss, axis=-1)


if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    tf.compat.v1.executing_eagerly = True

    model = tf.keras.models.load_model(get_file_path(ENTIRE_MODEL_PATH),
                                       custom_objects={'loss': DirectionalMeanSquaredError()}, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=DirectionalMeanSquaredError())

    all_indexes = read_indexes_prices()

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(all_indexes, model)
    sys.exit(app.exec_())
