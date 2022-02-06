import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import (Input,
                                     Embedding,
                                     Dense,
                                     Layer,
                                     )
from tensorflow.keras.models import Model
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

        # 针对通道维执行标准化
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


class EncoderLayer(Layer):

    def __init__(self,
                 embedding_size=None,
                 multihead_num=None,
                 dropout=None,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.dropout = dropout
        self.attention = MultiHeadsAttention(embedding_size=self.embedding_size,
                                             multihead_num=self.multihead_num,
                                             dropout=self.dropout)

        self.feed_forward = FeedForwardLayer(embedding_size=self.embedding_size,
                                             dropout=self.dropout)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'multihead_num': self.multihead_num,
            'dropout': self.dropout
        })

        return config

    def call(self, enc_input, attention_mask=None, *args, **kwargs):
        enc_output, attention = self.attention([enc_input, enc_input, enc_input], attention_mask)

        enc_output = self.feed_forward(enc_output)

        return enc_output, attention


class Encoder(Layer):
    """
    编码器
    """
    def __init__(self,
                 embedding_size=None,
                 vocab_size=None,
                 max_seq_len=None,
                 multihead_num=None,
                 num_layers=None,
                 dropout=None,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.multihead_num = multihead_num
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_layers = [EncoderLayer(embedding_size=self.embedding_size,
                                            multihead_num=self.multihead_num,
                                            dropout=self.dropout) for _ in range(self.num_layers)]
        self.sequence_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size)
        self.position_encode = PositionalEncoding(max_seq_len=self.max_seq_len,
                                                  embedding_size=self.embedding_size)

    def get_config(self):

        config = super(Encoder, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'multihead_num': self.multihead_num,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        })

        return config

    def padding_mask(self, seq):

        pad_mask = tf.expand_dims(tf.cast(tf.equal(seq, 0), tf.float32), axis=-1)
        pad_mask = tf.multiply(pad_mask, tf.transpose(pad_mask, [0, 2, 1]))

        return pad_mask

    def call(self, inputs, *args, **kwargs):

        output = self.position_encode(self.sequence_embedding(inputs))
        self_attention_mask = self.padding_mask(inputs)
        enc_outputs, attentions = [], []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            enc_outputs.append(output)
            attentions.append(attention)

        return enc_outputs, attentions


class DecoderLayer(Layer):

    def __init__(self,
                 embedding_size=None,
                 multihead_num=None,
                 dropout=None,
                 **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.dropout = dropout
        self.attention1 = MultiHeadsAttention(embedding_size=self.embedding_size,
                                              multihead_num=self.multihead_num,
                                              dropout=self.dropout)

        self.attention2 = MultiHeadsAttention(embedding_size=self.embedding_size,
                                              multihead_num=self.multihead_num,
                                              dropout=self.dropout)

        self.feed_forward = FeedForwardLayer(embedding_size=self.embedding_size,
                                             dropout=self.dropout)

    def get_config(self):

        config = super(DecoderLayer, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'multihead_num': self.multihead_num,
            'dropout': self.dropout
        })

        return config

    def call(self, dec_input, enc_output=None, self_attention_mask=None, context_attention_mask=None):

        dec_output, self_attention = self.attention1([dec_input, dec_input, dec_input], self_attention_mask)

        dec_output, context_attention = self.attention2([dec_output, enc_output], context_attention_mask)

        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(Layer):
    """
    解码器
    """
    def __init__(self,
                 embedding_size=None,
                 vocab_size=None,
                 max_seq_len=None,
                 multihead_num=None,
                 num_layers=None,
                 dropout=None,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.multihead_num = multihead_num
        self.num_layers = num_layers
        self.dropout = dropout
        self.decoder_layers = [DecoderLayer(embedding_size=self.embedding_size,
                                            multihead_num=self.multihead_num,
                                            dropout=self.dropout) for _ in range(self.num_layers)]
        self.sequence_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size)
        self.position_encode = PositionalEncoding(max_seq_len=self.max_seq_len,
                                                  embedding_size=self.embedding_size)

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'multihead_num': self.multihead_num,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        })

        return config

    def padding_mask(self, seq):
        """
        标准文本掩码
        """
        pad_mask = tf.expand_dims(tf.cast(tf.equal(seq, 0), tf.float32), axis=-1)
        pad_mask = tf.multiply(pad_mask, tf.transpose(pad_mask, [0, 2, 1]))

        return pad_mask

    def sequence_mask(self, seq):
        """
        上三角掩码, 用于遮挡当前字符的后续文本
        保证同样的字符不会推理为不同的结果
        """
        mask = 1 - tf.linalg.band_part(tf.ones(shape=(tf.shape(seq)[1],) * 2), -1, 0)
        mask = tf.expand_dims(mask, axis=0)

        return mask

    def call(self, inputs, enc_outputs=None, context_attention_mask=None):
        assert enc_outputs
        output = self.position_encode(self.sequence_embedding(inputs))
        self_attention_padding_mask = self.padding_mask(inputs)
        seq_mask = self.sequence_mask(inputs)
        self_attention_mask = tf.cast(tf.greater(self_attention_padding_mask + seq_mask, 0), dtype=tf.float32)

        self_attentions, context_attentions = [], []
        for i, decoder in enumerate(self.decoder_layers):
            output, self_attention, context_attention = decoder(output, enc_outputs[i],
                                                                self_attention_mask, context_attention_mask)
            self_attentions.append(self_attention)
            context_attentions.append(context_attention)

        return output, self_attentions, context_attentions


class Transformer(Model):
    def __init__(self,
                 embedding_size=None,
                 src_vocab_size=None,
                 src_max_len=None,
                 tgt_vocab_size=None,
                 tgt_max_len=None,
                 multihead_num=None,
                 num_layers=None,
                 dropout=None,
                 **kwargs):
        """
        切勿于call中实例化layer对象
        """
        super(Transformer, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.src_vocab_size = src_vocab_size
        self.src_max_len = src_max_len
        self.tgt_vocab_size = tgt_vocab_size
        self.tgt_max_len = tgt_max_len
        self.multihead_num = multihead_num
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder = Encoder(embedding_size=self.embedding_size,
                               vocab_size=self.src_vocab_size,
                               max_seq_len=self.src_max_len,
                               multihead_num=self.multihead_num,
                               num_layers=self.num_layers,
                               dropout=self.dropout)
        self.decoder = Decoder(embedding_size=self.embedding_size,
                               vocab_size=self.tgt_vocab_size,
                               max_seq_len=self.tgt_max_len,
                               multihead_num=self.multihead_num,
                               num_layers=self.num_layers,
                               dropout=self.dropout)
        self.linear = Dense(units=self.tgt_vocab_size, kernel_initializer=initializers.GlorotNormal(),
                            kernel_regularizer=l2(5e-4), use_bias=False)

    def get_config(self):

        config = {}
        # config = super(Transformer, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'src_vocab_size': self.src_vocab_size,
            'src_max_len': self.src_max_len,
            'tgt_vocab_size': self.tgt_vocab_size,
            'tgt_max_len': self.tgt_max_len,
            'multihead_num': self.multihead_num,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        })
        return config

    def padding_mask(self, seq, tgt_size):

        pad_mask = tf.tile(tf.expand_dims(tf.cast(tf.equal(seq, 0), tf.float32), axis=1), [1, tgt_size, 1])
        # pad_mask = tf.multiply(pad_mask, tf.transpose(pad_mask, [0, 2, 1]))
        return pad_mask

    def call(self, inputs, training=None, mask=None):

        assert isinstance(inputs, list)

        src_seq, tgt_seq = inputs

        context_attention_mask = self.padding_mask(src_seq, tf.shape(tgt_seq)[1])  # src_seq, tgt_seq

        enc_outputs, _ = self.encoder(src_seq)
        output, _, _ = self.decoder(tgt_seq, enc_outputs, context_attention_mask)
        output = self.linear(output)

        return output


class MaskedSparseCategoricalCrossentropy(Loss):
    """
    自定会掩码误差
    """
    def __init__(self, **kwargs):
        super(MaskedSparseCategoricalCrossentropy, self).__init__(**kwargs)
        self.base_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, y_true, y_pred):

        loss = self.base_loss(y_true, y_pred)
        # 排除掩码误差
        masked_y_true = tf.cast(tf.not_equal(y_true, 0), dtype=loss.dtype)

        # masked_loss = tf.boolean_mask(loss, masked_y_true)
        masked_loss = loss * masked_y_true

        return tf.reduce_mean(masked_loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    自定义学习率
    """
    def __init__(self, d_model=512, warmup_steps=3072):  # warmup_steps=4000
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
