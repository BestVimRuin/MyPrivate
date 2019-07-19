import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r"D:\\gitee\\learn_tf\\re_learn\\data\\", one_hot=True)


# 第一层权重是784*500   第二层是500*10
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1), )

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 前向传播,
def inference(input_tensor, regularizer, reuse=False):
    # 假如每一层都写一个命名空间  几百层的也要写吗?, 在同一程序中多次调用,需将  reuse=True
    with tf.variable_scope('layer1',reuse=reuse):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE, ], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)


    with tf.variable_scope('layer2',reuse=reuse):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE, ], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    return layer2




