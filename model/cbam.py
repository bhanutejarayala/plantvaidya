"""
CBAM (Convolutional Block Attention Module)
=============================================
Implements channel + spatial attention, used on top of the MobileNetV2
backbone to focus the classifier on disease-lesion regions of the leaf.

This matches the architecture used in the fine-tuned MobileNetV2+CBAM
tomato leaf disease classifier (Journal of Engineering / Wiley submission).
"""

import tensorflow as tf
from tensorflow.keras import layers


def channel_attention(input_feature, ratio: int = 8):
    channel = input_feature.shape[-1]

    shared_dense_one = layers.Dense(channel // ratio, activation="relu",
                                     kernel_initializer="he_normal", use_bias=True)
    shared_dense_two = layers.Dense(channel, kernel_initializer="he_normal", use_bias=True)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    feature = layers.Add()([avg_pool, max_pool])
    feature = layers.Activation("sigmoid")(feature)

    return layers.Multiply()([input_feature, feature])


def spatial_attention(input_feature, kernel_size: int = 7):
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    feature = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=1,
                             padding="same", activation="sigmoid",
                             kernel_initializer="he_normal", use_bias=False)(concat)

    return layers.Multiply()([input_feature, feature])


def cbam_block(input_feature, ratio: int = 8, kernel_size: int = 7):
    """Apply channel attention followed by spatial attention (CBAM)."""
    x = channel_attention(input_feature, ratio)
    x = spatial_attention(x, kernel_size)
    return x
