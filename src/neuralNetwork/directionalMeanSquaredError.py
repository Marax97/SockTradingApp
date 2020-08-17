import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops


class DirectionalMeanSquaredError(tf.keras.losses.Loss):

    def __init__(self):
        super(DirectionalMeanSquaredError, self).__init__(
            name='directionalMeanSquaredError', reduction=losses_utils.ReductionV2.AUTO)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        pass_days = K.variable(y_true[:, :-1], dtype='float32')
        y_true = y_true[:, 1:]

        direction = tf.math.multiply(K.sign(y_true - pass_days), K.sign(y_pred - pass_days))
        direction = tf.math.pow(K.constant(0.5), direction)
        loss = tf.math.multiply(math_ops.squared_difference(y_pred, y_true), direction)
        return K.mean(loss, axis=-1)
