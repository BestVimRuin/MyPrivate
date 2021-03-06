import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from re_learn.mnist01 import mnist_inference
from re_learn.mnist01 import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y_input')

        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        corrent_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    print(ckpt.model_checkpoint_path)
                    saver.restore(sess, 'D:\\gitee\\learn_tf\\re_learn\\mnist01\\model\\model.ckpt-29001')
                    # 估计是windows的问题 '/' '\'的差异
                    # saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('第{}次, {}'.format(global_step, accuracy_score))
                else:
                    print('没有存储的模型')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets(r"D:\\gitee\\learn_tf\\re_learn\\data\\", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()