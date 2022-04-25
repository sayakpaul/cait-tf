"""
CaiT models in TensorFlow.

Reference:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cait.py
"""

from copy import deepcopy
from typing import List

import ml_collections as mlc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .layers import ClassAttn, LayerScale, StochasticDepth, TalkingHeadAttn


def mlp(x: int, dropout_rate: float, hidden_units: List[int]):
    """FFN for a Transformer block."""
    for (idx, units) in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=tf.nn.gelu if idx == 0 else None,
            bias_initializer=keras.initializers.RandomNormal(stddev=1e-6),
        )(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def LayerScaleBlockClassAttn(
    config: mlc.ConfigDict, drop_prob: float, name: str
):
    """Pre-norm transformer block meant to be applied to the embedding of the
    cls token and the embeddings of image patches.

    Includes LayerScale and Stochastic Depth.
    """
    x = keras.Input((None, config.projection_dim))
    x_cls = keras.Input((None, config.projection_dim))
    inputs = layers.Concatenate(axis=1)([x_cls, x])

    # Class attention (CA).
    x1 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(inputs)
    attn_output, attn_scores = ClassAttn(config)(x1)
    attn_output = (
        LayerScale(config)(attn_output) if config.init_values else attn_output
    )
    attn_output = (
        StochasticDepth(drop_prob)(attn_output) if drop_prob else attn_output
    )
    x2 = layers.Add()([x_cls, attn_output])

    # FFN.
    x3 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(x2)
    x4 = mlp(
        x3, hidden_units=config.mlp_units, dropout_rate=config.dropout_rate
    )
    x4 = LayerScale(config)(x4) if config.init_values else x4
    x4 = StochasticDepth(drop_prob)(x4) if drop_prob else x4
    outputs = layers.Add()([x2, x4])

    return keras.Model([x, x_cls], [outputs, attn_scores], name=name)


# class LayerScaleBlock(keras.Model):
#     """Pre-norm transformer block meant to be applied to the embeddings of the
#     image patches.

#     Includes LayerScale and Stochastic Depth.
#     """

#     def __init__(self, config: mlc.ConfigDict, drop_prob: float, **kwargs):
#         super().__init__(**kwargs)
#         self.config = config
#         self.drop_prob = drop_prob

#         self.norm1 = layers.LayerNormalization(epsilon=config.layer_norm_eps)
#         self.attn = TalkingHeadAttn(config)
#         self.ls1 = LayerScale(config) if config.init_values else tf.identity
#         self.dp1 = StochasticDepth(drop_prob) if drop_prob else tf.identity

#         self.norm2 = layers.LayerNormalization(epsilon=config.layer_norm_eps)
#         self.mlp_fn = partial(
#             mlp, hidden_units=config.mlp_units, dropout_rate=config.dropout_rate
#         )
#         self.ls2 = LayerScale(config) if config.init_values else tf.identity
#         self.dp2 = StochasticDepth(drop_prob) if drop_prob else tf.identity

#     def call(self, x):
#         # Self-attention between patches.
#         x1 = self.norm1(x)
#         attn_output, attn_scores = self.attn(x1)
#         attn_output = self.ls1(attn_output)
#         attn_output = self.dp1(attn_output)
#         x2 = x + attn_output

#         # FFN.
#         x3 = self.norm2(x2)
#         x4 = self.mlp_fn(x3)
#         x4 = self.ls2(x4)
#         x4 = self.dp2(x4)
#         outputs = x2 + x4

#         return outputs, attn_scores


def LayerScaleBlock(config: mlc.ConfigDict, drop_prob: float, name: str):
    """Pre-norm transformer block meant to be applied to the embeddings of the
    image patches.

    Includes LayerScale and Stochastic Depth.
    """
    encoded_patches = layers.Input((None, config.projection_dim))

    # Self-attention.
    x1 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(
        encoded_patches
    )
    attn_output, attn_scores = TalkingHeadAttn(config)(x1)
    attn_output = (
        LayerScale(config)(attn_output) if config.init_values else attn_output
    )
    attn_output = (
        StochasticDepth(drop_prob)(attn_output) if drop_prob else attn_output
    )
    x2 = layers.Add()([encoded_patches, attn_output])

    # FFN.
    x3 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(x2)
    x4 = mlp(
        x3, hidden_units=config.mlp_units, dropout_rate=config.dropout_rate
    )
    x4 = LayerScale(config)(x4) if config.init_values else x4
    x4 = StochasticDepth(drop_prob)(x4) if drop_prob else x4
    outputs = layers.Add()([x2, x4])

    return keras.Model(encoded_patches, [outputs, attn_scores], name=name)


class CaiT(keras.Model):
    """CaiT model."""

    def __init__(self, config: mlc.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=config.projection_dim,
                    kernel_size=(config.patch_size, config.patch_size),
                    strides=(config.patch_size, config.patch_size),
                    padding="VALID",
                    name="conv_projection",
                    kernel_initializer="lecun_normal",
                ),
                layers.Reshape(
                    target_shape=(-1, config.projection_dim),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )

        self.cls_token = tf.Variable(tf.zeros((1, 1, config.projection_dim)))
        self.pos_embed = tf.Variable(
            tf.zeros((1, config.num_patches, config.projection_dim))
        )

        self.pos_drop = layers.Dropout(
            config.dropout_rate, name="projection_dropout"
        )

        dpr = [config.drop_path_rate for _ in range(config.sa_ffn_layers)]

        self.blocks = [
            LayerScaleBlock(config, name=f"sa_ffn_block_{i}", drop_prob=dpr[i])
            for i in range(config.sa_ffn_layers)
        ]

        ca_config = deepcopy(config)
        with ca_config.unlocked():
            ca_config.dropout_rate = 0.0
        self.blocks_token_only = [
            LayerScaleBlockClassAttn(
                config=ca_config, name=f"ca_ffn_block_{i}", drop_prob=0.0
            )
            for i in range(config.ca_ffn_layers)
        ]

        self.norm = layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="head_norm"
        )

        if config.pre_logits:
            self.head = layers.Dense(
                config.num_classes, name="classification_head"
            )

    def call(self, x):
        x = self.projection(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # SA+FFN layers.
        sa_ffn_attn = {}
        for blk in self.blocks:
            x, attn_scores = blk(x)
            sa_ffn_attn[f"{blk.name}_att"] = attn_scores

        # CA+FFN layers.
        ca_ffn_attn = {}
        cls_tokens = tf.tile(self.cls_token, (tf.shape(x)[0], 1, 1))
        for blk in self.blocks_token_only:
            cls_tokens, attn_scores = blk([x, cls_tokens])
            ca_ffn_attn[f"{blk.name}_att"] = attn_scores

        x = tf.concat([cls_tokens, x], axis=1)
        x = self.norm(x)

        # Always return the attention scores from the SA+FFN and CA+FFN layers
        # for convenience.
        if self.config.global_pool:
            x = (
                tf.reduce_mean(x[:, 1:], axis=1)
                if self.config.global_pool == "avg"
                else x[:, 0]
            )
        return (
            (x, sa_ffn_attn, ca_ffn_attn)
            if self.config.pre_logits
            else (self.head(x), sa_ffn_attn, ca_ffn_attn)
        )
