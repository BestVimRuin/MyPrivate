import tensorflow as tf
import numpy as np

def get_weight(shape, lamda):
    # 生成一个变量,权重
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)

    # 好像是把这个变量加入到新的collection中
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamda)(var))
    return var

# 这个是真实值~~~
x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='x_input')
# 这个是真实值~~~
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')
batch_size = 8
# 定义每层网络节点个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时的最深的节点, 开始的时候是输入层
cur_layer = x
# 当前层的网络节点的个数
in_dimension = layer_dimension[0]


# 循环生成5层神经网络
for i in range(1, n_layers): # 这个为啥不n_layers+1
    # 此处为下一层的网络节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量, 并将这个变量的L2正则化损失假如计算图上的集
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

    cur_layertf = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]


# 在定义神经网络前向传播的同时,已经将所有的L2正则化损失加入了图上的集合
# 这里只需要计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方误差损失函数假如损失集合
tf.add_to_collection('losses', mse_loss)
# get_collection返回一个列表, 这个列表是所有这个集合中的元素, 这个样例中,
# 这些元素就是损失函数的不同部分, 将他们加起来就可以得到最终的损失函数
loss = tf.add_n(tf.get_collection('losses'))







