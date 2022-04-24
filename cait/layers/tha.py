"""
Reference:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cait.py
"""

import ml_collections as mlc
import tensorflow as tf
from tensorflow.keras import layers


class TalkingHeadAttn(layers.Layer):
    def __init__(self, config: mlc.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.num_heads = self.config.num_heads

        head_dim = self.projection_dim // self.num_heads

        self.scale = head_dim ** -0.5

        self.qkv = layers.Dense(config.projection_dim * 3)
        self.attn_drop = layers.Dropout(config.dropout_rate)

        self.proj = layers.Dense(config.projection_dim)

        self.proj_l = layers.Dense(config.num_heads)
        self.proj_w = layers.Dense(config.num_heads)

        self.proj_drop = layers.Dropout(config.dropout_rate)

    def call(self, x, training):
        B, N, C = tf.shape(x)
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]

        attn = tf.matmul(q, k, transpose_b=True)
        attn = self.proj_l(tf.tranpose(attn, perm=[0, 2, 3, 1]))
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = tf.nn.softmax(attn, axis=-1)

        attn = self.proj_w(tf.transpose(attn, perm=[0, 2, 3, 1]))
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x, training)
        return x
