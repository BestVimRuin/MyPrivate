# -*- coding=utf-8 -*-
import pickle as pk
import tensorflow as tf
import os
import argparse
import warnings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s-[%(levelname)s]: - %(message)s')
logger = logging.getLogger(__name__)

logger.info('[AI-MAP] DeepFMNormalPredict')

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
embed_size = info.input3
hidden_layers = info.input4
hidden_units = info.input5
num_classes = info.input6
test_size = info.input7
dropout = info.input8
lr = info.input9
l2 = info.input10
model_file_path = info.input11
result_hdfs_path = info.input12
keytab_file = info.input13

result_flag = info.output0

# 参数类型转换
try:
    batch_size = int(batch_size)
except:
    logger.warning("[AI-MAP]-param：batch_size transition failed，use default value：1024")
    batch_size = 1024

try:
    embed_size = int(embed_size)
except:
    logger.warning("[AI-MAP]-param：embed_size transition failed，use default value：5")
    embed_size = 5

try:
    hidden_layers = int(hidden_layers)
except:
    logger.warning("[AI-MAP]-param：hidden_layers transition failed，use default value：3")
    hidden_layers = 3

try:
    hidden_units = hidden_units.strip().split(',')
    if len(hidden_units) != 3:
        logger.warning("[AI-MAP]-param：hidden_units transition failed，use default value：[1024, 512, 256]")
        hidden_units = [1024, 512, 256]
except:
    logger.warning("[AI-MAP]-param：hidden_units transition failed，use default value：[1024, 512, 256]")
    hidden_units = [1024, 512, 256]

try:
    num_classes = int(num_classes)
except:
    logger.warning("[AI-MAP]-param：num_classes transition failed，use default value：2")
    num_classes = 2

try:
    test_size = float(test_size)
except:
    logger.warning("[AI-MAP]-param：test_size transition failed，use default value：0.2")
    test_size = 0.2

try:
    dropout = float(dropout)
except:
    logger.warning("[AI-MAP]-param：header transition failed，use default value：None")
    dropout = 0.5

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
logger.info('[AI-MAP]-embed_size: %s' % embed_size)
logger.info('[AI-MAP]-hidden_layers: %s' % hidden_layers)
logger.info('[AI-MAP]-hidden_units: %s' % hidden_units)
logger.info('[AI-MAP]-num_classes: %s' % num_classes)
logger.info('[AI-MAP]-test_size: %s' % test_size)
logger.info('[AI-MAP]-dropout: %s' % dropout)
logger.info('[AI-MAP]-lr: %s' % lr)
logger.info('[AI-MAP]-l2: %s' % l2)
logger.info('[AI-MAP]-model_file_path: %s' % model_file_path)
logger.info('[AI-MAP]-result_hdfs_path: %s' % result_hdfs_path)
logger.info('[AI-MAP]-keytab_file: %s' % keytab_file)


# test_file 类型：string 默认值： 无 说明： 测试的数据文件
# batch_size = 512  # 类型：int；默认值：1024；参数说明：mini-batch梯度下降算法的batch大小，越大训练波动越小，训练收敛所需时间越长
# embed_size = 5  # 类型：int；默认值：5；参数说明：embedding向量维度，维度越高模型计算量越大，拟合越准确
# hidden_layers = 3  # 类型：int；默认值：3；参数说明：DNN模型部分隐藏层层数
# hidden_units = [1024, 512, 256]  # 类型：list；默认值：[1024, 512, 256]；参数说明：DNN模型部分隐藏层每层神经元个数，list大小需要与hidden_layers参数对应
# num_classes = 2  # 类型：int；默认值：2；参数说明：类别数目，2分类问题值为2
# test_size = 0.2  # 类型：float；默认值：0.2；参数说明：测试样本比例
# dropout = 0.75  # 类型：float；默认值：0.5；参数说明：DNN模型部分dropout层保留数据概率
# lr = 0.001  # 类型：float；默认值：0.001；参数说明：梯度下降算法学习率，学习率越大，参数更新幅度越大，越不容易收敛
# l2 = 0.001  # 类型：float；默认值：0.001；参数说明：L2正则项权重，L2正则项防止过拟合，权重越高，拟合度越低
# model_file_path 训练模型保存的结果
# result_hdfs_path 模型测试结果保存地址
# keytab_file  类型：string 默认值：无 说明：kerberos认证用户的秘钥


class DataSetsTest(object):

    def __init__(self, data_path):
        logger.info('[AI-MAP]-load test data')
        feature_value_columns, feature_index_columns, feature_size, test_data = pk.load(open(data_path, "rb"))

        self.feature_size = feature_size
        self.field_size = len(feature_index_columns)

        feature_index = test_data[feature_index_columns].values
        feature_value = test_data[feature_value_columns].values

        key_cols = ['client_id', 'prdt_code']
        self.user_prdt = test_data[key_cols]
        self.test = DataSet(feature_index, feature_value)
        del feature_value_columns
        del feature_index_columns
        del test_data


class DataSet(object):

    def __init__(self, feature_index, feature_value):
        self.num = len(feature_index)
        self.feature_index = feature_index
        self.feature_value = feature_value
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
        feature_index = self.feature_index[start:end]
        feature_value = self.feature_value[start:end]
        return feature_index, feature_value


class DFM(object):

    def __init__(self, data_sets):
        self.data_sets = data_sets

        self.field_size = self.data_sets.field_size
        self.feature_size = self.data_sets.feature_size

        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            self.embeddings_fea = tf.get_variable(name="embeddings_fe",
                                                initializer=tf.random_normal([self.feature_size, embed_size], 0,
                                                                             0.01))
        with tf.variable_scope("weight", reuse=tf.AUTO_REUSE):
            self.bias_fea = tf.get_variable(name="weight_fe",
                                          initializer=tf.random_uniform([self.feature_size, 1], 0.0, 1.0))
        self.add_placeholders()
        self.output = self.inference()

    def add_placeholders(self):
        """
        add data placeholder to graph
        """
        self.feature_index_placeholder = tf.placeholder(tf.int32, shape=(None, self.field_size))
        self.feature_value_placeholder = tf.placeholder(tf.float32, shape=(None, self.field_size))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def inference(self):
        feature_value = tf.reshape(self.feature_value_placeholder, shape=[-1, self.field_size, 1])
        self.y_first_order = tf.nn.embedding_lookup(self.bias_fea, self.feature_index_placeholder)
        self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feature_value), 2)
        self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_placeholder)  # None * F

        self.embeddings = tf.nn.embedding_lookup(self.embeddings_fea, self.feature_index_placeholder)
        self.embeddings = tf.multiply(self.embeddings, feature_value)

        summed_features_emb_square = tf.square(tf.reduce_sum(self.embeddings, 1))  # None * K
        squared_sum_features_emb = tf.reduce_sum(tf.square(self.embeddings), 1)  # None * K

        self.y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
        self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_placeholder)

        self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * embed_size])
        for i in range(hidden_layers):
            self.y_deep = tf.layers.dense(self.y_deep, hidden_units[i], activation=tf.nn.relu, name="h" + str(i),
                                          reuse=tf.AUTO_REUSE)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_placeholder)

        output = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
        output = tf.layers.dense(output, 2, activation=None, name="output", reuse=tf.AUTO_REUSE)
        output = tf.nn.softmax(output)
        return output

    def dense_batch(self, inputs, is_train, scope):
        if is_train:
            h = tf.layers.batch_normalization(inputs, axis=0, training=is_train, name=scope)
        else:
            h = tf.layers.batch_normalization(inputs, axis=0, training=is_train, name=scope, reuse=True)
        return tf.nn.relu(h, 'relu')

    def predict(self, session):
        predictions = []
        while self.data_sets.test.has_next():
            feature_index, feature_value = self.data_sets.test.next_batch()
            feed_dict = {
                self.feature_index_placeholder: feature_index,
                self.feature_value_placeholder: feature_value,
                self.dropout_placeholder: 1
            }
            ps = session.run(self.output, feed_dict=feed_dict)
            # output是一个二维的
            predictions += [p[0] for p in ps]
        return predictions


if __name__ == "__main__":
    logger.info("[AI-MAP]-START DeepFMNormalPredict Module!")
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
        model = DFM(data_set_test)
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
    logger.info("[AI-MAP]-END NormalDeepFMModelPredict Module!")