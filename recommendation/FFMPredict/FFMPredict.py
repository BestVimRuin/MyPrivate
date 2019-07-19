# -*- coding=utf-8 -*-
import pickle as pk
import tensorflow as tf
import numpy as np
import os
import argparse
import warnings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s-[%(levelname)s]: - %(message)s')
logger = logging.getLogger(__name__)

logger.info('[AI-MAP] FFMModePredict')

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
parser.add_argument('--input13', type=str)

parser.add_argument('--output0', type=str)

info = parser.parse_args()
run_flag = info.input0
test_file = info.input1
batch_size = info.input2
vector_dimension = info.input3
test_size = info.input4
lr = info.input5
l2 = info.input6
model_file_path = info.input7
result_hdfs_path = info.input8
keytab_file = info.input9

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
    vector_dimension = 5

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

logger.info('[AI-MAP]-test_file: %s' % test_file)
logger.info('[AI-MAP]-batch_size: %s' % batch_size)
logger.info('[AI-MAP]-vector_dimension: %s' % vector_dimension)
logger.info('[AI-MAP]-test_size: %s' % test_size)
logger.info('[AI-MAP]-lr: %s' % lr)
logger.info('[AI-MAP]-l2: %s' % l2)
logger.info('[AI-MAP]-model_file_path: %s' % model_file_path)
logger.info('[AI-MAP]-result_hdfs_path: %s' % result_hdfs_path)
logger.info('[AI-MAP]-keytab_file: %s' % keytab_file)


# test_file 类型：string 默认值： 无 说明： 测试的数据文件
# batch_size = 512  # 类型：int；默认值：1024；参数说明：mini-batch梯度下降算法的batch大小，越大训练波动越小，训练收敛所需时间越长
# vector_dimension = 5  # 类型：int；默认值：5；参数说明：v向量维度，维度越高模型计算量越大，拟合越准确
# test_size = 0.2  # 类型：float；默认值：0.2；参数说明：测试样本比例
# lr = 0.001  # 类型：float；默认值：0.001；参数说明：梯度下降算法学习率，学习率越大，参数更新幅度越大，越不容易收敛
# l2 = 0.001  # 类型：float；默认值：0.001；参数说明：L2正则项权重，L2正则项防止过拟合，权重越高，拟合度越低
# model_file_path 训练模型保存的结果
# result_hdfs_path 模型测试结果保存地址
# keytab_file  类型：string 默认值：无 说明：kerberos认证用户的秘钥


class DataSetsTest(object):

    def __init__(self, data_path):
        logger.info('[AI-MAP]-load test data')
        value_columns, feature_size, x_field, field_size, data_test = pk.load(open(test_file, "rb"))
        self.feature_size = feature_size
        self.field_size = field_size
        x_test = data_test[value_columns]

        key_cols = ['client_id', 'prdt_code']
        self.user_prdt = data_test[key_cols]
        self.test = DataSet(x_test)
        del x_field
        del value_columns
        del data_test


class DataSet(object):

    def __init__(self, x_test):
        self.num = len(x_test)
        self.x_test = x_test
        self.epoch_completed = 0
        self.index_in_epoch = 0

    def has_next(self):
        if self.index_in_epoch > self.num:
            return False
        return True

    def next_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        end = min(self.index_in_epoch, self.num)
        x_test = self.x_test[start:end]
        return x_test


class FFM(object):

    def __init__(self, data_sets):
        self.data_sets = data_sets
        self.feature_size = self.data_sets.feature_size
        self.field_size = self.data_sets.field_size

        with tf.variable_scope("linear_weight", reuse=tf.AUTO_REUSE):
            self.w0 = tf.get_variable(name='bias', initializer=tf.constant(0.01),dtype=np.float32)
            self.w = tf.get_variable(name='w', initializer=tf.random_normal([self.feature_size],0.0,1.0))
        with tf.variable_scope('cross_pair_v'):
            self.v = tf.get_variable(name='v',
                                     initializer=tf.random_normal([self.feature_size, self.field_size, vector_dimension],
                                                                  mean=0, stddev=0.01))

        self.add_placeholders()
        self.output = self.inference()
    def add_placeholders(self):
        """add data_ori placeholder to graph"""
        self.x = tf.placeholder(tf.float32, shape=(None, None))
        self.y = tf.placeholder(tf.float32, shape=(None, 1))

    def inference(self):
        # model
        # linear part
        linear_terms = tf.reshape(tf.add(self.w0, tf.reduce_sum(tf.multiply(self.w, self.x), 1)),shape=[-1,1])  # n * 1

        # ------- pair order term ----------
        pair_interactions = tf.constant(0, dtype='float32')
        self.x_field = self.data_sets.x_field
        for i in range(self.feature_size):
            for j in range(i + 1, self.feature_size):
                pair_interactions += tf.multiply(
                    tf.reduce_sum(tf.multiply(self.v[i, self.x_field[i]], self.v[j, self.x_field[j]])),
                    tf.multiply(self.x[:, i], self.x[:, j])
                )
            # shape of [None, 1]
        self.pair_interactions = tf.reshape(pair_interactions, shape=(-1, 1))
        output = tf.nn.sigmoid(tf.add(linear_terms,self. pair_interactions))
        return output

    def predict(self, session):
        predictions = []
        while self.data_sets.test.has_next():
            x_test = self.data_sets.test.next_batch()
            feed_dict = {
                self.x: x_test
            }
            ps = session.run(self.output, feed_dict=feed_dict)
            # output是一个二维的
            predictions += [p[0] for p in ps]
        return predictions


if __name__ == "__main__":
    logger.info("[AI-MAP]-START FMModlePredict Module!")
    # kerberos authentication
    # keytab = keytab_file.strip('\n').strip().split('/')[-1]
    user = keytab_file.strip('\n').strip().split('/')[-1].split('.')[0]
    command = "kinit -kt %s %s" % (keytab_file, user)
    print(command)
    cmd_status = os.system(command)
    if cmd_status != 0:
        raise Exception("COMMAND : %s " % command, "FAILED!!!")

    with tf.Graph().as_default():
        data_set_test = DataSetsTest(test_file)
        model = FFM(data_set_test)
        saver = tf.train.Saver()
        with tf.Session() as session:
            # 重新载入最后一次训练的模型结果
            saver.restore(session, tf.train.latest_checkpoint(model_file_path))
            predictions = model.predict(session)
            logger.info("[AI-MAP]-len(predictions)", len(predictions))
            data_set_test.user_prdt['label'] = predictions
            logger.info('[AI-MAP]-save the predict result to file: %s' % result_hdfs_path)
            result_file_name = test_file.split('/')[-1].replace('sample', 'result')
            data_set_test.user_prdt.to_csv(result_file_name, index=False, encoding='utf-8')

    # 上传到指定hdfs地址
    command = 'export HADOOP_USER_NAME=u006586;hadoop fs -rm %s' % result_hdfs_path + result_file_name
    print(command)
    os.system(command)
    command = 'export HADOOP_USER_NAME=u006586;hadoop fs -put %s %s' % (result_file_name, result_hdfs_path)
    print(command)
    cmd_status = os.system(command)
    if cmd_status != 0:
        raise Exception("COMMAND : %s " % command, "FAILED!!!")

    with open(result_flag, 'w') as f_r:
        f_r.write('True')
    logger.info("[AI-MAP]-END FMModelPredict Module!")