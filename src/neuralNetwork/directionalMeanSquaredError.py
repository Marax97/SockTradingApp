import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import losses_utils


class DirectionalMeanSquaredError(tf.keras.losses.Loss):

    def __init__(self, height, width):
        super(DirectionalMeanSquaredError, self).__init__(
            name='directionalMeanSquaredError', reduction=losses_utils.ReductionV2.AUTO)
        self.height = height
        self.width = width

    def call(self, y_true, y_pred):
        zeros = K.variable(K.zeros([self.height, self.width], dtype='float32'))
        zeros[:, 1:].assign(y_true[:, :-1])

        direction = tf.math.multiply(K.sign(y_true - zeros), K.sign(y_pred - zeros))
        direction = tf.math.scalar_mul(K.constant(-2.0), direction)
        loss = tf.math.multiply(tf.math.squared_difference(y_pred, y_true), direction)
        return K.mean(loss, axis=-1)
