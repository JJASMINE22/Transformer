# -*- coding: UTF-8 -*-
'''
@Project ：Transformer
@File    ：CustomLayers.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers


class LayerNormalization(Layer):
    """
    层归一化, 主要应用于自然语言处理
    """
    def __init__(self,
                 epsilon=1e-6,
                 center=True,
                 scale=True,
                 gamma_initializer=initializers.Ones(),
                 beta_initializer=initializers.Zeros(),
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer
        self.gamma_regularizer = gamma_regularizer
        self.beta_regularizer = beta_regularizer
        self.gamma_constraint = gamma_constraint
        self.beta_constraint = beta_constraint

    def build(self, input_shape):

        super(LayerNormalization, self).build(input_shape)

        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint,
                                     trainable=True if self.scale or self.center else False,
                                     name='gamma')

        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint,
                                    trainable=True if self.scale or self.center else False,
                                    name='beta')

        self.built

    def call(self, inputs, *args, **kwargs):

        # 针对通道维执行标准化, 降低多样性
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        var = tf.reduce_sum(tf.math.pow((inputs - mean), 2), axis=-1, keepdims=True) / inputs.shape[-1]
        normalize = tf.math.divide((inputs - mean), tf.sqrt(var + self.epsilon))

        if tf.logical_or(self.center, self.scale):
            normalize = tf.nn.bias_add(tf.multiply(self.gamma, normalize), self.beta)

        return normalize


class PositionalEncoding(Layer):
    """
    位置编码
    """
    def __init__(self,
                 max_seq_len=None,
                 embedding_size=None,
                 **kwargs):
        """
        :param max_seq_len: 最大输入文本长度
        :param embedding_size: 嵌入维度
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.position_encoding = np.array([[pos / 10000 ** (2 * (i // 2) / self.embedding_size)
                                            for i in range(self.embedding_size)]
                                           for pos in range(self.max_seq_len)])
        # 三角编码
        self.position_encoding[:, 0::2] = tf.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = tf.cos(self.position_encoding[:, 1::2])
        # 此处的位置编码矩阵为numpy array, 若该层处于支持可形变输入的Model对象时, 需转换为tensor array
        self.position_encoding = self.position_encoding[tf.newaxis, ...]

    def get_config(self):

        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'embedding_size': self.embedding_size
        })

        return config

    def call(self, inputs, *args, **kwargs):

        seq_len = inputs.shape[1]
        inputs *= tf.sqrt(tf.cast(self.embedding_size, dtype=tf.float32))
        # 显示图像, 发现特征随位置由小→大
        # 多数人认为seq_len由输入文本长度决定, 然而文本不定长, seq_len实际上与字典尺寸一致
        inputs += self.position_encoding[:, :seq_len]

        return inputs

class MultiHeadsAttention(Layer):
    """
    多头注意力
    """

    def __init__(self,
                 embedding_size=None,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.Zeros(),
                 kernel_regularizer=l2(5e-4),
                 bias_regularizer=l2(5e-4),
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_attention_bias=False,
                 use_attention_activation=False,
                 multihead_num=None,
                 dropout=None,
                 **kwargs):
        super(MultiHeadsAttention, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_attention_bias = use_attention_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        self.use_attention_activation = use_attention_activation
        self.multihead_num = multihead_num
        self.dropout = dropout
        self.layernorm = LayerNormalization()

    def get_config(self):

        config = super(MultiHeadsAttention, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'use_attention_bias': self.use_attention_bias,
            'bias_initializer': self.bias_initializer,
            'bias_regularizer': self.bias_regularizer,
            'bias_constraint': self.bias_constraint,
            'use_attention_activation': self.use_attention_activation,
            'multihead_num': self.multihead_num,
            'dropout': self.dropout
        })

        return config

    def build(self, input_shape):

        self.kernel_q = self.add_weight(shape=(input_shape[0][-1], self.embedding_size),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_q')

        self.kernel_k = self.add_weight(shape=(input_shape[1][-1], self.embedding_size),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_k')

        self.kernel_v = self.add_weight(shape=(input_shape[-1][-1], self.embedding_size),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_v')

        self.kernel = self.add_weight(shape=(input_shape[0][-1], self.embedding_size),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      name='kernel')

        if self.use_attention_bias:
            self.attention_bias = self.add_weight(shape=(1,),
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint,
                                                  trainable=True,
                                                  name='attention_bias')

        self.built = True

    def call(self, inputs, mask=None, *args, **kwargs):
        """
        编码器中q、k、v均为src输入源特征
        解码器中q为tgt目标特征, k与v均为src输入源特征
        """
        assert 2 <= len(inputs) < 4

        q = tf.matmul(inputs[0], self.kernel_q)
        k = tf.matmul(inputs[1], self.kernel_k)
        v = tf.matmul(inputs[-1], self.kernel_v) if len(inputs) == 3 \
            else tf.matmul(inputs[1], self.kernel_v)
        seq_len_q, seq_len_k, seq_len_v = q.shape[1], k.shape[1], v.shape[1]

        # q = tf.reshape(q, shape=[-1, seq_len_q, self.embedding_size//self.multihead_num])
        # k = tf.reshape(k, shape=[-1, seq_len_k, self.embedding_size//self.multihead_num])
        # v = tf.reshape(v, shape=[-1, seq_len_v, self.embedding_size//self.multihead_num])

        # 区别与原文的reshape, 此处用split→concat实现多头, 并还原
        q = tf.concat(tf.split(q, num_or_size_splits=self.multihead_num, axis=-1), axis=0)
        k = tf.concat(tf.split(k, num_or_size_splits=self.multihead_num, axis=-1), axis=0)
        v = tf.concat(tf.split(v, num_or_size_splits=self.multihead_num, axis=-1), axis=0)

        attention = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.embedding_size // self.multihead_num,
                                                                        dtype=tf.float32))
        if self.use_attention_bias:
            attention += self.attention_bias

        if self.use_attention_activation:
            attention = tf.nn.tanh(attention)

        if mask is not None:
            # mask = tf.repeat(mask, self.multihead_num, axis=0)
            mask = tf.tile(mask, [self.multihead_num, 1, 1])
            attention -= 1e+9 * mask

        attention = tf.nn.softmax(attention)

        context = tf.matmul(attention, v)

        context = tf.concat(tf.split(context, num_or_size_splits=self.multihead_num, axis=0), axis=-1)
        # context = tf.reshape(context, shape=[-1, seq_len_q, self.embedding_size])

        output = tf.matmul(context, self.kernel)

        output = tf.nn.dropout(output, rate=self.dropout)

        output = tf.add(inputs[0], output)

        output = self.layernorm(output)

        return output, attention


class FeedForwardLayer(Layer):

    def __init__(self,
                 embedding_size=None,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.Zeros(),
                 kernel_regularizer=l2(5e-4),
                 bias_regularizer=l2(5e-4),
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_bias=True,
                 activation=True,
                 dropout=None,
                 **kwargs):
        super(FeedForwardLayer, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.use_bias = use_bias
        self.activation = activation
        self.dropout = dropout
        self.layernorm = LayerNormalization()

    def get_config(self):

        config = super(FeedForwardLayer, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'dropout': self.dropout
        })

        return config

    def build(self, input_shape):

        self.inner_kernel = self.add_weight(shape=(self.embedding_size, self.embedding_size * 4),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint,
                                            trainable=True,
                                            name='inner_kernel')

        self.outter_kernel = self.add_weight(shape=(self.embedding_size * 4, self.embedding_size),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             constraint=self.kernel_constraint,
                                             trainable=True,
                                             name='outter_kernel')

        if self.use_bias:

            self.inner_bias = self.add_weight(shape=(self.embedding_size * 4, ),
                                              initializer=self.bias_initializer,
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint,
                                              trainable=True,
                                              name='inner_bias')

            self.outter_bias = self.add_weight(shape=(self.embedding_size, ),
                                               initializer=self.bias_initializer,
                                               regularizer=self.bias_regularizer,
                                               constraint=self.bias_constraint,
                                               trainable=True,
                                               name='outter_bias')

        self.built = True

    def call(self, inputs, *args, **kwargs):

        x = tf.matmul(inputs, self.inner_kernel)

        if self.use_bias:
            x += self.inner_bias

        if self.activation:
            x = tf.nn.relu(x)

        x = tf.matmul(x, self.outter_kernel)

        if self.use_bias:
            x += self.outter_bias

        x = tf.nn.dropout(x, rate=self.dropout)

        x = tf.add(inputs, x)

        x = self.layernorm(x)

        return x
