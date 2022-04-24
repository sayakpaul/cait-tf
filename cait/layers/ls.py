import ml_collections as mlc
import tensorflow as tf
from tensorflow.keras import layers


class LayerScale(layers.Layer):
    def __init__(self, config: mlc.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(
            config.init_values * tf.ones((config.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma
