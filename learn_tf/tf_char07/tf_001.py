import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 生成整数型属性
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets(r"D:\\gitee\\learn_tf\\re_learn\\data\\", dtype=tf.uint8,   one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出tfrecord文件的地址
filename = 'D:\\gitee\\learn_tf\\re_learn\\data\\output.tfrecords'
# 创建一个writer来写tfrecord文件
writer =tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    # 将图像转化成一个字符串
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels':_int64_feature(pixels),
        'labels':_int64_feature(np.argmax(labels[index])),
        'image_raw':_bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()




