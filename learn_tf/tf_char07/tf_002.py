import tensorflow as tf

reader = tf.TFRecordReader()

# queue队列，梯子, 默认shuffle=True为乱序
filename_queue = tf.train.string_input_producer(['D:\\gitee\\learn_tf\\re_learn\\data\\output.tfrecords'])

# serialize 序列化
_, serialized_example = reader.read(filename_queue)
print(_, '~~~~~~~~~~~~~~')
print(serialized_example, '############')

features = tf.parse_single_example(
    serialized_example,
    # 这个读出来的必须和写进去的key保持一致
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'labels': tf.FixedLenFeature([], tf.int64),
    }
)
image = tf.decode_raw(features['image_raw'], tf.uint8)
pixels = tf.cast(features['pixels'], tf.int32)
label = tf.cast(features['labels'], tf.int32)

sess = tf.Session()
coord = tf.train.Coordinator()
# 收集所有线程
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    print(sess.run([image, label, pixels]))