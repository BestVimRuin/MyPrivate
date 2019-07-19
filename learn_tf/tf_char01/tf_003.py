import tensorflow as tf
import numpy as np

batch_size = 8
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 这个是真实值~~~
x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='x_input')
# 这个是真实值~~~
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')

a = tf.matmul(x, w1)
# 这个才是预测的
y = tf.matmul(a, w2)

y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean\
    (y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = np.random.RandomState(1)
dataset_size = 128

# [128, 2]矩阵
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]



with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

    # 训练轮数
    STEPS = 5000
    for i in range(STEPS+1):
        # 每次选取batch_size个样本训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练,并更新参数
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})

            print("训练{}次, 交叉熵为{}".format(i, total_cross_entropy))

            print(sess.run(w1))
            print(sess.run(w2))




    # print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))