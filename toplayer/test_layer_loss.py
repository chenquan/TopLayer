from unittest import TestCase
import numpy as np
import tensorflow as tf

from toplayer.loss import layer_loss


class TestLayer_loss(TestCase):
    def test_layer_loss(self):
        x_data = np.array([[1., 2., 3.], [3., 2., 6.]])
        tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)
        print(tensor)
        loss = layer_loss(tensor, tensor, tensor)
        print(loss)
