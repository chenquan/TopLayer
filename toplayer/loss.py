import tensorflow as tf

from toplayer.sklearn import LarsLayer


@LarsLayer(name='loss', path="model/lars.m")
def layer_loss(X, y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
