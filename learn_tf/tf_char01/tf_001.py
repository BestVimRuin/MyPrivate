import tensorflow as tf



a = tf.constant([1., 2.], )
b = tf.constant([2., 3.], )
result = a + b
# print(result)
#
# sess = tf.Session()
# qqq = sess.run(result)



sess = tf.Session()
with sess.as_default():
    print(result.eval())