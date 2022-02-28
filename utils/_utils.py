# -*- coding: UTF-8 -*-
'''
@Project ：Transformer
@File    ：_utils.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import tensorflow as tf
from tensorflow.keras.losses import Loss

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
