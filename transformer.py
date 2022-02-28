# -*- coding: UTF-8 -*-
'''
@Project ：Transformer
@File    ：transformer.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from net.networks import Transformer
from utils._utils import CustomSchedule, MaskedSparseCategoricalCrossentropy

tf.config.run_functions_eagerly(True)
# 用于可变形状特征的训练加速
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
                        ]

test_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
                      ]


class TransFormer():
    def __init__(self,
                 src_vocab_size: int,
                 src_max_len: int,
                 tgt_vocab_size: int,
                 tgt_max_len: int,
                 learning_rate: float):
        """
        :param src_vocab_size: 输入文本字典尺寸
        :param src_max_len: 最大输入文本长度
        :param tgt_vocab_size: 目标文本字典尺寸
        :param tgt_max_len: 最大目标文本长度
        :param learning_rate: 学习率
        """

        self.model = Transformer(embedding_size=512,
                                 src_vocab_size=src_vocab_size,
                                 src_max_len=src_max_len,
                                 tgt_vocab_size=tgt_vocab_size,
                                 tgt_max_len=tgt_max_len,
                                 multihead_num=8,
                                 num_layers=6,
                                 dropout=0.3)

        self.learnig_rate = learning_rate

        self.loss_func = MaskedSparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=CustomSchedule(d_model=128),
                                                  beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.train_loss = tf.keras.metrics.Mean()
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        self.test_loss = tf.keras.metrics.Mean()
        self.test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function(input_signature=train_step_signature)
    def train(self, sources, logits, targets):

        with tf.GradientTape() as tape:
            predictions = self.model([sources, logits])
            loss = self.loss_func(targets, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_acc(targets, predictions)

    @tf.function(input_signature=test_step_signature)
    def test(self, sources, logits, targets):

        with tf.GradientTape() as tape:
            predictions = self.model([sources, logits])
            loss = self.loss_func(targets, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.test_loss(loss)
        self.test_acc(targets, predictions)
