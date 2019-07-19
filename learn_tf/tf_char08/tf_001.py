import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


HIDDEN_SIZE = 30
NUM_LATERS = 2
TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


def generate_date(seq):
    X = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        X.append(([seq[i:i + TIMESTEPS]]))
        y.append(([seq[i + TIMESTEPS]]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LATERS)])

    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    if not is_training:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
    return predictions, loss, train_op


def train(sess, train_X, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model'):
        predictions, loss, train_op = lstm_model(X, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print('第{}步：损失为：{}'.format(i, l))

def run_eval(sess,test_X,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope('model',reuse=True):
        predictions, _, _ = lstm_model(X, [0.0], False)

    predictions=[]
    labels= []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([predictions,y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels=np.array(labels).squeeze()
    rmse = np.sqrt(((predictions-labels)**2).mean(axis=0))
    print('均方误差：{}'.format(rmse))

    plt.figure()
    plt.plot(predictions,label='predictions')
    plt.plot(labels,label='real_sin')
    plt.legend()
    plt.show()


test_start = (TRAINING_EXAMPLES+TRAINING_STEPS)*SAMPLE_GAP
test_end = test_start+ (TESTING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP

train_X,train_y = generate_date(np.sin(np.linspace(test_start,test_end,TESTING_EXAMPLES+TIMESTEPS,dtype=np.float32)))


with tf.Session() as sess:
    train(sess, train_X,train_y)