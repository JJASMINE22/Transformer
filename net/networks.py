# -*- coding: UTF-8 -*-
'''
@Project ：Transformer
@File    ：networks.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import tensorflow as tf
from tensorflow.keras.layers import (Layer,
                                     Dense,
                                     Embedding)
from tensorflow.keras.models import Model
from CustomLayers import (MultiHeadsAttention,
                          FeedForwardLayer,
                          PositionalEncoding)

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