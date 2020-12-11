#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:48:50 2020
代码参考：
@author:https://github.com/dragen1860/Deep-Learning-with-TensorFlow-book

"""


import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, losses, optimizers

batchsz = 128 # 批量大小
total_words = 10000 # 词汇表大小N_vocab
max_review_len = 80 # 句子最大长度s，大于的句子部分将截断，小于的将填充
embedding_len = 100 # 词向量特征长度f
# 加载IMDB数据集，此处的数据采用数字编码，一个数字代表一个单词
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)


# 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
# 构建数据集，打散，批量，并丢掉最后一个不够batchsz的batch
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)


#LSTM
class MyRNN(keras.Model):

    def __init__(self, units):
        super(MyRNN, self).__init__()


        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # [b, 80, 100] , h_dim: 64
        self.rnn = keras.Sequential([
            # layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            # layers.SimpleRNN(units, dropout=0.5, unroll=True)

            layers.LSTM(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.LSTM(units, dropout=0.5, unroll=True)
        ])


        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training : bool = True):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # x: [b, 80, 100] => [b, 64]
        x = self.rnn(x)

        # out: [b, 64] => [b, 1]
        x = self.outlayer(x)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob





def main():
    units = 64 # RNN状态向量长度f
    epochs = 10 # 训练epochs
    training = True
    
    import time
    t0 = time.time()
    model = MyRNN(units)

    if training:
        # 装配
        model.compile(optimizer = optimizers.RMSprop(0.001),
                      loss = losses.BinaryCrossentropy(),
                      metrics=['accuracy'],
                      experimental_run_tf_function = False)
    else:
        # 装配
        model.compile(optimizer = optimizers.RMSprop(0.001),
          loss = losses.BinaryCrossentropy(),
          metrics=['accuracy'])

   

    # 训练和验证
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    # 测试
    model.evaluate(db_test)
    t1 = time.time()
    print("LSTM运行的时间为：{0:.2f} 秒".format(t1 - t0))
    

if __name__ == '__main__':
    main()
