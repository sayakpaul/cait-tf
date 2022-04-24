"""
Reference:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cait.py
"""

import ml_collections as mlc
import tensorflow as tf
from tensorflow.keras import layers


class ClassAttn(layers.Layer):
    def __init__(self, config: mlc.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        head_dim = config.projection_dim // config.num_heads
        self.scale = head_dim ** -0.5

        self.q = layers.Dense(config.projection_dim)
        self.k = layers.Dense(config.projection_dim)
        self.v = layers.Dense(config.projection_dim)
        self.attn_drop = layers.Dropout(config.dropout_rate)
        self.proj = layers.Dense(config.projection_dim)
        self.proj_drop = layers.Dropout(config.dropout_rate)

    def call(self, x, training):
        B, N, C = tf.shape(x)

        # Query projection. `cls_token` embeddings are queries.
        q = tf.expand_dims(self.q(x[:, 0]), axis=1)
        q = tf.reshape(
            q, (B, 1, self.config.num_heads, C // self.config.num_heads)
        )
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        scale = tf.cast(self.scale, dtype=q.dtype)
        q = q * scale

        # Key projection. Patch embeddings as well the cls embedding are used as keys.
        k = self.k(x)
        k = tf.reshape(
            k, (B, N, self.config.num_heads, C // self.config.num_heads)
        )
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        # Value projection. Patch embeddings as well the cls embedding are used as keys.
        v = self.v(x)
        v = tf.reshape(
            v, (B, N, self.config.num_heads, C // self.config.num_heads)
        )
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Calculate attention between cls_token embedding and patch embeddings.
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_cls = tf.matmul(attn, v)
        x_cls = tf.transpose(x_cls, perm=[0, 2, 1, 3])
        x_cls = tf.reshape(x_cls, (B, 1, C))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls, training)

        return x_cls, attn
