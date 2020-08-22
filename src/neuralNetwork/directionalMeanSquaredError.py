import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops


class DirectionalMeanSquaredError(tf.keras.losses.Loss):

    def __init__(self, passed_days=6, days_to_predict=4):
        super(DirectionalMeanSquaredError, self).__init__(
            name='directionalMeanSquaredError', reduction=losses_utils.ReductionV2.AUTO)
        self.passed_days = passed_days
        self.days_to_predict = days_to_predict

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        y_true = K.squeeze(y_true, axis=2)
        y_pred = K.squeeze(y_pred, axis=2)

        directions = self.fillDirections(y_true, y_pred)

        y_true = y_true[:, self.passed_days:]

        loss = math_ops.squared_difference(y_pred, y_true)
        loss = loss * directions

        return K.mean(loss, axis=-1)

    def fillDirections(self, y_true, y_pred):
        directions = list()
        stop = (self.passed_days + 1) / 7
        x = K.arange(0, stop, stop / (self.passed_days + 1), dtype=tf.float32)
        xm = K.mean(x, axis=0)

        for i in range(self.days_to_predict):
            y_t = y_true[:, i:self.passed_days + i + 1]

            y_p = K.concatenate((y_true[:, i:self.passed_days + i], y_pred[:, i:i + 1]), axis=1)

            y_t_m = K.repeat_elements(K.expand_dims(K.mean(y_t, 1), axis=1), rep=self.passed_days + 1, axis=1)
            y_p_m = K.repeat_elements(K.expand_dims(K.mean(y_t, 1), axis=1), rep=self.passed_days + 1, axis=1)

            slope_t = K.sum((y_t - y_t_m) * (x - xm), axis=1) / K.sum(K.pow(x - xm, 2))
            slope_p = K.sum((x - xm) * (y_p - y_p_m), axis=1) / K.sum(K.pow(x - xm, 2))
            directions.append(K.expand_dims(self.calculateLossBySlopesDirection(slope_t, slope_p)))
        return K.concatenate(directions, axis=1)

    def calculateLossBySlopesDirection(self, slope_true, slope_pred):
        directions = K.abs(slope_true - slope_pred)
        return directions
