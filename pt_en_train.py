# -*- coding: UTF-8 -*-
'''
@Project ：Transformer
@File    ：pt_en_train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''

import tensorflow_datasets as tfds
import tensorflow as tf
from transformer import TransFormer
from warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler

tf.config.run_functions_eagerly(True)


if __name__ == '__main__':

    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

    tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

    sample_string = 'Transformer is awesome.'

    tokenized_string = tokenizer_en.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_en.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    assert original_string == sample_string

    BUFFER_SIZE = 20000
    BATCH_SIZE = 64


    def encode(lang1, lang2):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
            lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            lang2.numpy()) + [tokenizer_en.vocab_size + 1]

        return lang1, lang2


    MAX_LENGTH = 40


    def filter_max_length(x, y, max_length=MAX_LENGTH):

        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)


    def tf_encode(pt, en):

        result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en


    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # 将数据集缓存到内存中以加快读取速度。
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

    checkpoint_path = ".\\pt_en_models\\checkpoints"

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # 原始字典中, 无起始符与未知符, 字典尺寸需加上
    transformer = TransFormer(src_vocab_size=tokenizer_pt.vocab_size + 2,
                              src_max_len=tokenizer_pt.vocab_size + 2,
                              tgt_vocab_size=tokenizer_en.vocab_size + 2,
                              tgt_max_len=tokenizer_en.vocab_size + 2,
                              learning_rate=1e-4,
                              resume_train=1e-5,
                              model_path=1e-7)

    ckpt = tf.train.Checkpoint(transformer=transformer.model,
                               optimizer=transformer.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    Epochs = 200

    train_file = open('.\\logs\\train_file.txt', 'w')
    test_file = open('.\\logs\\test_file.txt', 'w')
    train_file.write('loss' + ',' + 'accuracy\n')
    test_file.write('loss' + ',' + 'accuracy\n')

    for epoch in range(Epochs):

        for (batch, (inp, tar)) in enumerate(train_dataset):

            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            transformer.train(inp, tar_inp, tar_real)
            if not (batch+1) % 50:
                print(transformer.train_loss.result())

        for (batch, (inp, tar)) in enumerate(val_dataset):

            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            transformer.test(inp, tar_inp, tar_real)
            if not (batch+1) % 50:
                print(transformer.test_loss.result())

        print(
            f'Epoch {epoch + 1}, '
            f'train_loss: {transformer.train_loss.result()}, '
            f'train_acc: {transformer.train_acc.result()*100}, '
            f'test_loss: {transformer.test_loss.result()}, '
            f'test_acc: {transformer.test_acc.result()*100}'
        )

        log = '.\\tf_models\\Epoch{:0>3d}_train_loss{:.3f}_test_loss{:.3f}.h5'.format(epoch + 1,
                                                                                      transformer.train_loss.result(),
                                                                                      transformer.test_loss.result())

        transformer.model.save_weights(log)

        train_file.write('{:.3f},{:.3f}\n'.format(transformer.train_loss.result(),
                                                  transformer.train_acc.result()*100))
        test_file.write('{:.3f},{:.3f}\n'.format(transformer.test_loss.result(),
                                                 transformer.test_acc.result()*100))

        # 于下次迭代开始前清空记录
        transformer.train_loss.reset_states()
        transformer.train_acc.reset_states()
        transformer.test_loss.reset_states()
        transformer.test_acc.reset_states()