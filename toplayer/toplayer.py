import abc
from functools import wraps

import numpy as np
import tensorflow as tf

__all__ = [
    "TopLayer"
]


class TopLayer(object, metaclass=abc.ABCMeta):
    """
    The decorator base class for TensorFlow's Keras
    advanced API top-level external to other frameworks.

    :parameter
    name: The name of the current layer
    path:
    """

    def __init__(self, name, path):
        self.name = name
        self.path = path

    def __call__(self, loss_func):
        """

        :param loss_func: Wrapper function
        :return: The path to the TopLayer model.
        """
        model = self.model
        name = self.name
        path = self.path

        @wraps(loss_func)
        def wrapper(x, y_true, y_pred):
            with tf.Session() as sess:
                x_value, y_true_v = np.array(sess.run([x, y_true]))
                # print("x_value", x_value.shape)
                # print("y_true_v", y_true_v.shape)
                y_pred_v, = model(x_value, y_true_v, path)
                y_pred = tf.convert_to_tensor(y_pred_v, name=name)
                return loss_func(x, y_true, y_pred)

        return wrapper

    @abc.abstractmethod
    def model(self, x, y_true_v, path):
        """
        :param x: Output layer data.
        :param y_true_v: Output layer actual value.
        :param path: The path to the TopLayer model.
        :return: Output layer prediction value
        """
        raise NotImplemented
