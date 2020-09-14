import tensorflow as tf
import sys
import matplotlib
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from pickle import load
import random

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

ENTIRE_MODEL_PATH = '\\resources\\savedModel'
SCALER_TRAIN_PATH = '\\resources\\savedModel\\scalers\\ScalerTrain.pkl'
SCALER_TEST_PATH = '\\resources\\savedModel\\scalers\\ScalerTest.pkl'

INDEXES_DIRECTORY = "\\resources\\stockPricesWithIndicators\\"
SELECTED_COLUMNS = "\\resources\\SelectedColumns.csv"
SYMBOLS_FILE = "\\resources\\symbols.csv"

TARGET_COLUMN = ['Adj Close']


def load_selected_features():
    df = pd.read_csv(get_file_path(SELECTED_COLUMNS), sep='\s*,\s*', engine='python')
    return df['SelectedColumns'].astype(str).values.tolist()


def read_indexes_prices():
    files = utils.get_files_in_directory(INDEXES_DIRECTORY)

    index_prices = {}
    for file in files:
        index = pd.read_csv(utils.get_file_path(INDEXES_DIRECTORY + file))
        index.set_index("Date", inplace=True)
        index = index[load_selected_features()]
        index_prices[file[:file.rfind(".")]] = index
    return index_prices


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=10, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, indexes, model):
        super().__init__()
        self.indexes = indexes
        self.model = model

        self.canvas = MplCanvas(self)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.init_select_index(indexes))
        self.data = pd.DataFrame([])
        self.on_predict()
        layout.addWidget(self.canvas)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowTitle('Stock trading app')
        self.update_plot()
        self.show()

        # Timer to redraw the canvas
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        self.canvas.axes.cla()  # Clear the canvas.
        self.data.plot(ax=self.canvas.axes)
        # # Trigger the canvas to update and redraw.
        self.canvas.draw()

    def on_predict(self):
        selected_day = self.date_calendar.date().toString("yyyy-MM-dd")
        selected_index = self.indexes.get(self.index_combo.currentText().split(" - ")[1])
        datetime.strptime(selected_day, "%Y-%m-%d")
        past_days = selected_index[:selected_day].iloc[-30:]
        future_days = selected_index[selected_day:].iloc[:5]

        prediction = self.model.predict(numpy.array([scaler_train.transform(past_days)]))[0]
        prediction = scaler_test.inverse_transform(prediction)
        prediction = numpy.insert(prediction, 0, future_days[TARGET_COLUMN][:1])

        self.data = pd.DataFrame(index=past_days.index.append(future_days.index[1:]))
        self.data["Past days"] = past_days[TARGET_COLUMN]
        self.data["Real feature days"] = future_days[TARGET_COLUMN]

        self.data["Predicted days"] = pd.DataFrame(index=future_days.index, data=prediction)

    def init_select_index(self, indexes):
        gridLayout = QtWidgets.QGridLayout()

        self.index_label = QtWidgets.QLabel(self)
        self.index_label.setText("Select index: ")
        gridLayout.addWidget(self.index_label, 1, 1)
        self.index_combo = QtWidgets.QComboBox(self)
        self.index_combo.addItems(symbols.apply(
            lambda x: ' - '.join(x.dropna().astype(str)),
            axis=1
        ))
        gridLayout.addWidget(self.index_combo, 1, 2)
        symbols
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
        gridLayout.addWidget(self.predict_button, 3, 1, 1, 2)

        grid_widget = QtWidgets.QWidget()
        grid_widget.setLayout(gridLayout)
        grid_widget.setMaximumWidth(300)
        grid_widget.setMaximumHeight(100)
        return grid_widget


if __name__ == "__main__":
    model = tf.keras.models.load_model(get_file_path(ENTIRE_MODEL_PATH),
                                       custom_objects={'loss': DirectionalMeanSquaredError()}, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=DirectionalMeanSquaredError())

    scaler_train = load(open(get_file_path(SCALER_TRAIN_PATH), 'rb'))
    scaler_test = load(open(get_file_path(SCALER_TEST_PATH), 'rb'))
    all_indexes = read_indexes_prices()
    symbols = pd.read_csv(utils.get_file_path(SYMBOLS_FILE), sep='\s*,\s*', engine='python')

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(all_indexes, model)
    sys.exit(app.exec_())
