from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r"D:\\gitee\\learn_tf\\re_learn\\data\\", one_hot=True)

print('训练数据大小:', mnist.train.num_examples)

print('验证数据大小:', mnist.validation.num_examples)

print('测试数据大小:', mnist.test.num_examples)

print('样本数据:', mnist.train.images[0])

print('样本数据标签:', mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
