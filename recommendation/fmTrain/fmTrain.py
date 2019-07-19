# -*- coding=utf-8 -*-
import pickle as pk
import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import classification_report, roc_curve, auc
import tensorflow as tf
import argparse
import warnings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s-[%(levelname)s]: - %(message)s')
logger = logging.getLogger(__name__)

logger.info('[AI-MAP] DeepFMNormalTrain')

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--input0', type=str)
parser.add_argument('--input1', type=str)
parser.add_argument('--input2', type=str)
parser.add_argument('--input3', type=str)
parser.add_argument('--input4', type=str)
parser.add_argument('--input5', type=str)
parser.add_argument('--input6', type=str)
parser.add_argument('--input7', type=str)
parser.add_argument('--input8', type=str)
parser.add_argument('--input9', type=str)
parser.add_argument('--input10', type=str)
parser.add_argument('--input11', type=str)
parser.add_argument('--input12', type=str)

parser.add_argument('--output0', type=str)

info = parser.parse_args()
run_flag = info.input0
train_file = info.input1
eval_file = info.input2
batch_size = info.input3
vector_dimension = info.input4
test_size = info.input5
lr = info.input6
l2 = info.input7
max_train = info.input8
optimizer_type = info.input9
model_path = info.input10
evaluation_path = info.input11
keytab_file = info.input12

result_flag = info.output0

# 参数类型转换
try:
    batch_size = int(batch_size)
except:
    logger.warning("[AI-MAP]-param：batch_size transition failed，use default value：1024")
    batch_size = 1024

try:
    vector_dimension = int(vector_dimension)
except:
    logger.warning("[AI-MAP]-param：embed_size transition failed，use default value：5")
    batch_size = 5


try:
    test_size = float(test_size)
except:
    logger.warning("[AI-MAP]-param：test_size transition failed，use default value：0.2")
    test_size = 0.2

try:
    lr = float(lr)
except:
    logger.warning("[AI-MAP]-param：lr transition failed，use default value：0.001")
    lr = 0.001

try:
    l2 = float(l2)
except:
    logger.warning("[AI-MAP]-param：l2 transition failed，use default value：0.001")
    l2 = 0.001

try:
    max_train = int(max_train)
except:
    logger.warning("[AI-MAP]-param：max_train transition failed，use default value：100000")
    max_train = 100000

if optimizer_type not in ['adam', 'adagrad', 'gd', 'momentum']:
    logger.warning("[AI-MAP]-param：optimizer_type transition failed，use default value：adam")
    optimizer_type = 'adam'

logger.info('[AI-MAP]-train_file: %s' % train_file)
logger.info('[AI-MAP]-eval_file: %s' % eval_file)
logger.info('[AI-MAP]-batch_size: %s' % batch_size)
logger.info('[AI-MAP]-vector_dimension: %s' % vector_dimension)
logger.info('[AI-MAP]-test_size: %s' % test_size)
logger.info('[AI-MAP]-lr: %s' % lr)
logger.info('[AI-MAP]-l2: %s' % l2)
logger.info('[AI-MAP]-max_train: %s' % max_train)
logger.info('[AI-MAP]-optimizer_type: %s' % optimizer_type)
logger.info('[AI-MAP]-model_path: %s' % model_path)
logger.info('[AI-MAP]-evaluation_path: %s' % evaluation_path)
logger.info('[AI-MAP]-keytab_file: %s' % keytab_file)


# train_file 类型：string 默认值： 无 说明： 训练数据文件
# eval_file 类型：string 默认值： 无 说明： 训练数据文件
# batch_size = 512  # 类型：int；默认值：1024；参数说明：mini-batch梯度下降算法的batch大小，越大训练波动越小，训练收敛所需时间越长
# vector_dimension = 5  # 类型：int；默认值：5；参数说明：vector_dimension向量维度，维度越高模型计算量越大，拟合越准确
# hidden_layers = 3  # 类型：int；默认值：3；参数说明：DNN模型部分隐藏层层数
# test_size = 0.2  # 类型：float；默认值：0.2；参数说明：测试样本比例
# lr = 0.001  # 类型：float；默认值：0.001；参数说明：梯度下降算法学习率，学习率越大，参数更新幅度越大，越不容易收敛
# l2 = 0.001  # 类型：float；默认值：0.001；参数说明：L2正则项权重，L2正则项防止过拟合，权重越高，拟合度越低
# max_train 类型： int 默认值： 100000 说明： 训练最大次数
# optimizer_type = 'adam'  # 类型： string 默认值： adam 说明：optimizer 优化器类型
# model_path 模型结果存放的目录
# evaluation_path evaluation_path结果存放地址
# keytab_file  类型：string 默认值：无 说明：kerberos认证用户的秘钥

class DataSetsTrain():
    '''
    Define the Train  DataSet, contain the train and evaluation data set.
    '''

    def __init__(self, train_file, eval_file):
        print(train_file)
        print(eval_file)
        print(os.path.getsize(train_file))
        print(os.path.getsize(eval_file))
        print(os.listdir('.'))

        index_columns, value_columns, feature_size, data = pk.load(open(train_file, "rb"))
        _, _, _, data_eval = pk.load(open(eval_file, "rb"))

        print(data.shape)
        print(data_eval.shape)

        perm = np.arange(len(data))
        np.random.shuffle(perm)
        fea_index = data[index_columns].values[perm]
        fea_value = data[value_columns].values[perm]
        labels = data["label"].values[perm]

        self.train = DataSet(fea_index, fea_value, labels)

        fea_index_eval = data_eval[index_columns].values
        fea_value_eval = data_eval[value_columns].values
        labels_eval = data_eval["label"].values
        self.eval = DataSet(fea_index_eval, fea_value_eval, labels_eval)

        self.feature_size = feature_size

        print("train - positive: %d, negative: %d" % (sum(data["label"] == 1), sum(data["label"] == 0)))
        print("eval - positive: %d, negative: %d" % (sum(data_eval["label"] == 1), sum(data_eval["label"] == 0)))

        del data
        del data_eval
        del index_columns
        del value_columns


class DataSet(object):
    '''
    Define the DataSet, which defines the data batch for train or eval.
    '''
    def __init__(self, fea_index, fea_value, labels):
        '''
        Initialization.
        '''
        self.num = len(fea_index)
        self.fea_index = fea_index
        self.fea_value = fea_value
        self.labels = labels
        self.epoch_completed = 0
        self.index_in_epoch = 0

    def next_batch(self):
        '''
        Get next batch with defined batch size.
        '''
        if self.epoch_completed == 0 and self.index_in_epoch == 0:
            perm = np.arange(self.num)
            np.random.shuffle(perm)
            self.fea_index = self.fea_index[perm]
            self.fea_value = self.fea_value[perm]
            self.labels = self.labels[perm]
        if self.index_in_epoch + batch_size > self.num:
            # get rest untrained data and combine
            self.epoch_completed += 1
            rest_num = self.num - self.index_in_epoch
            fea_index_rest = self.fea_index[self.index_in_epoch:]
            fea_value_rest = self.fea_value[self.index_in_epoch:]
            labels_rest = self.labels[self.index_in_epoch:]

            perm = np.arange(self.num)
            np.random.shuffle(perm)
            self.fea_index = self.fea_index[perm]
            self.fea_value = self.fea_value[perm]
            self.labels = self.labels[perm]
            self.index_in_epoch = batch_size - rest_num
            fea_index_new = self.fea_index[:self.index_in_epoch]
            fea_value_new = self.fea_value[:self.index_in_epoch]
            labels_new = self.labels[:self.index_in_epoch]

            fea_index = np.concatenate((fea_index_rest, fea_index_new), axis=0)
            fea_value = np.concatenate((fea_value_rest, fea_value_new), axis=0)
            labels = np.concatenate((labels_rest, labels_new), axis=0)
            return fea_index, fea_value, labels
        else:
            start = self.index_in_epoch
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            fea_index = self.fea_index[start:end]
            fea_value = self.fea_value[start:end]
            labels = self.labels[start:end]
            return fea_index, fea_value, labels

class FM(object):
    """
    FM模型代码
    """
    def __init__(self, data_sets):
        self.data_sets = data_sets
        self.feature_size = self.data_sets.feature_size
        with tf.variable_scope("weight", reuse=tf.AUTO_REUSE):
            self.w0 = tf.get_variable(name='bias', initializer=tf.constant(0.01),dtype=np.float32)
            self.w = tf.get_variable(name='w', initializer=tf.random_normal([self.feature_size],0.0,1.0),)
            self.v = tf.get_variable(name='v', initializer=tf.random_normal([self.feature_size, vector_dimension], mean=0, stddev=0.01))

        self.add_placeholders()
        self.output = self.inference()
        self.accuracy = self.get_accuracy(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)

    def add_placeholders(self):
        """add data_ori placeholder to graph"""
        self.index = tf.placeholder(tf.int32, shape=(None, None))
        self.value = tf.placeholder(tf.float32, shape=(None, None))
        self.y = tf.placeholder(tf.float32, shape=(None, 1))

    def inference(self):
        # model
        # linear part
        self.w_i = tf.nn.embedding_lookup(self.w, self.index)
        linear_terms = tf.reshape(tf.add(self.w0, tf.reduce_sum(tf.multiply(self.w_i, self.value), 1)),shape=[-1,1])  # n * 1

        # ------- pair order term ----------
        embedding = tf.nn.embedding_lookup(self.v, self.index)
        feature_value = tf.reshape(self.value, shape=[-1, tf.shape(self.index)[1], 1])
        embeddings = tf.multiply(embedding, feature_value)
        pair_interactions = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.square(tf.reduce_sum(embeddings, axis=1)),
                tf.reduce_sum(tf.square(embeddings), axis=1)
            ), axis=1, keep_dims=True)
        output = tf.nn.sigmoid(tf.add(linear_terms, pair_interactions))
        return output


    def get_accuracy(self, output):
        threshold_value = tf.multiply(tf.ones(tf.shape(output)), tf.constant([0.5]))
        output_ = tf.concat([output, threshold_value], axis=1)
        y_ = tf.concat([self.y, threshold_value], axis=1)
        correct_pred = tf.equal(tf.argmax(output_, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
        return accuracy
    def add_loss_op(self, output):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.y))
        for vi in tf.trainable_variables():
            if not 'bias' in vi.name.lower():
                loss += l2 * tf.nn.l2_loss(vi)
        return loss

    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        # optimizer
        if optimizer_type == "adam":
            return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)
        elif optimizer_type == "adagrad":
            return tf.train.AdagradOptimizer(learning_rate=lr, initial_accumulator_value=1e-8).minimize(loss)
        elif optimizer_type == "gd":
            return tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        elif optimizer_type == "momentum":
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.95).minimize(loss)

    def train(self, session, saver, max_epochs, model_path):
        '''
        模型训练，并将模型结果保存至 model_path下，隐含矩阵保存在 embedding_path下。
        '''
        losses = []
        his_auc = [0]
        epoch_completed = self.data_sets.train.epoch_completed
        for epoch in range(max_epochs):
            s_time = time.time()
            fea_index, fea_value, label_ = self.data_sets.train.next_batch()

            label = np.reshape(label_.astype(np.float32), (-1,1))
            feed_dict = {
                self.index: fea_index,
                self.value: fea_value,
                self.y: label
            }
            loss, _, acr = session.run([self.calculate_loss, self.train_step, self.accuracy], feed_dict=feed_dict)
            losses.append(loss)
            duration = time.time() - s_time
            if epoch_completed != self.data_sets.train.epoch_completed or epoch == max_epochs - 1:
                epoch_completed = self.data_sets.train.epoch_completed
                auc_val = self.do_eval(session, self.data_sets.eval, epoch)
                # auc is larger than the max history auc, then save to result
                if auc_val > max(his_auc):
                    saver.save(session, model_path + "model.ckpt", global_step=epoch)
                his_auc.append(auc_val)
                # auc is less and less, then break
                if len(his_auc) > 5:
                    if his_auc[-1] < his_auc[-2] and his_auc[-2] < his_auc[-3] and his_auc[-3] < his_auc[-4] and \
                            his_auc[-4] < his_auc[-5]:
                        break
            if epoch % 100 == 0 or epoch == max_epochs - 1:
                print('Step %d: loss = %.2f, accuracy = %.2f (%.3f sec)' % (epoch, loss, acr / batch_size, duration))

        return losses
    def do_eval(self, session, data_set, epoch):
        '''
        Do evaluation for input date_set, and return the AUC value
        '''
        index = data_set.index_in_epoch
        data_set.index_in_epoch = 0
        epochs = data_set.num // batch_size
        total_num = epochs * batch_size
        acc_num = 0
        predictions = []
        labels = []
        output = self.inference()
        accuracy = self.get_accuracy(output)
        for step in range(epochs):
            fea_index, fea_value, label_ = data_set.next_batch()
            label = np.reshape(label_.astype(np.float32), (-1, 1))
            feed_dict = {
                self.index: fea_index,
                self.value: fea_value,
                self.y: label
            }
            ps, acc = session.run([output, accuracy], feed_dict=feed_dict)
            acc_num += acc
            predictions += [p[0] for p in ps]
            labels += [l[0] for l in label]
        df_pre = pd.DataFrame(predictions)
        eval_result_file = 'eval_%d' % epoch
        df_pre.to_csv('eval_%d' % epoch, header=None, index=False)
        del df_pre

        # 上传到指定hdfs地址
        command = 'export HADOOP_USER_NAME=u006586;hadoop fs -rm %s' % evaluation_path + eval_result_file
        print(command)
        os.system(command)
        command = 'export HADOOP_USER_NAME=u006586;hadoop fs -put %s %s' % (eval_result_file, evaluation_path)
        print(command)
        os.system(command)

        data_set.index_in_epoch = index
        if sum((data_set.labels == 0).astype(np.float32)) > 0:
            fpr, tpr, _ = roc_curve(labels, predictions, pos_label=1)
            au = auc(fpr, tpr)
        else:
            au = -1
        print('Data Set: %d(%d P, %d N)  Accuracy @ 1: %0.04f  AUC: %0.04f' % (
        data_set.num, sum((data_set.labels == 1).astype(np.float32)), sum((data_set.labels == 0).astype(np.float32)), acc_num / total_num, au))
        print(classification_report(labels, [1 if p > 0.5 else 0 for p in predictions]))
        return au

if __name__ == "__main__":
    logger.info("[AI-MAP]-START FMTrain Module!")
    s_t = time.time()
    user = keytab_file.strip('\n').strip().split('/')[-1].split('.')[0]
    command = "kinit -kt %s %s" % (keytab_file, user)
    print(command)
    cmd_status = os.system(command)
    if cmd_status != 0:
        raise Exception("COMMAND : %s " % command, "FAILED!!!")
    print(train_file, '\t', eval_file, '\t', model_path)
    with tf.Graph().as_default():
        df_train = DataSetsTrain(train_file, eval_file)
        model = FM(df_train)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            losses = model.train(session, saver, max_train, model_path)

    with open(result_flag, 'w') as f_r:
        f_r.write('True')
    e_t = time.time()
    print('cost time:', e_t-s_t)
    logger.info("[AI-MAP]-END FMTrain Module!")