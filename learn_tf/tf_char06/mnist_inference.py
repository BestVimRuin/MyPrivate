import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r"D:\\gitee\\learn_tf\\re_learn\\data\\", one_hot=True)

# 第一层权重是784*500   第二层是500*10
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
LAYER1_NODE = 500

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第一层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全链接层节点个数
FC_SIZE = 512


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1), )

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 前向传播,
def inference(input_tensor, train, regularizer, reuse=False):
    # 假如每一层都写一个命名空间  几百层的也要写吗?, 在同一程序中多次调用,需将  reuse=True
    with tf.variable_scope('layer1_conv1', reuse=reuse):
        conv1_weights = tf.get_variable('weights', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层最大池化层
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 神经网络第三层，卷积第二层
    with tf.variable_scope('layer3_conv2'):
        conv2_weights = tf.get_variable('weights', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第二层最大池化层
    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool2.get_shape().as_list()

    # pool_shape[0]是batch中数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5_fcl'):
        fcl_weights = tf.get_variable('weights', [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全链接层的权重需要正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fcl_weights))
        fcl_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.truncated_normal_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fcl_weights) + fcl_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)


    with tf.variable_scope('layer6_fc2'):
        fc2_weights = tf.get_variable('weights', [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全链接层的权重需要正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.truncated_normal_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weights)

    return logit
