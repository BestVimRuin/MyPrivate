import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from re_learn.tf_char06 import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY =  0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = 'D:\\gitee\\learn_tf\\re_learn\\mnist01\\model\\'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    # mnist_inference.IMAGE_SIZE主要是因为图片是28*28的
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                                    mnist_inference.NUM_CHANNELS], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y_input')



    # REGULARIZATION_RATE  公式中的lamda
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    # 前向传播
    y = mnist_inference.inference(x, train, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 滑动平均(不理解)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 滑动平均(不理解)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # argmax返回每行的最大值的索引
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 学习速率指数衰减, mnist.train.num_examples/BATCH_SIZE表示一共有多少轮
    learning_rate = tf.train.exponential_decay \
        (LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    # 梯度下降训练目标
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,
                                         mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})

            if i % 1000 == 0:
                print('第{}次, 损失值为 : {}.'.format(step ,loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets(r"D:\\gitee\\learn_tf\\re_learn\\data\\", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()




