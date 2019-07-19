import tensorflow as tf

input1 = tf.constant([1.,2.,],name='input1')
input2 = tf.Variable(tf.random_uniform([2]),name='input2')
output = tf.add_n([input1,input2],name='output1')

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    print(sess.run(output))

writer = tf.summary.FileWriter('./log',tf.get_default_graph())
writer.close()