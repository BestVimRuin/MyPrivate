# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os
import codecs
import pickle as pk
import argparse
import warnings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s-[%(levelname)s]: - %(message)s')
logger = logging.getLogger(__name__)

logger.info('[AI-MAP]  GenDfmSampleNormal')

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

parser.add_argument('--output0', type=str)

info = parser.parse_args()
run_flag = info.input0
train_sample_path = info.input1
client_feas_col_path = info.input2
prdt_feas_col_path = info.input3
client_feas_col_number_path = info.input4
mean_std_path = info.input5
discrete_path = info.input6
quantile_path = info.input7
train_flag = info.input8
number_flag = info.input9
result_train_sample_path = info.input10
keytab_file = info.input11

result_flag = info.output0

try:
    train_flag = int(train_flag)
except:
    logger.warning("[AI-MAP]-param：train_flag transition failed，use default value：1")
    train_flag = 1

try:
    number_flag = int(number_flag)
except:
    logger.warning("[AI-MAP]-param：number_flag transition failed，use default value：1")
    number_flag = 1

logger.info('[AI-MAP]-train_sample_path: %s' % train_sample_path)
logger.info('[AI-MAP]-client_feas_col_path: %s' % client_feas_col_path)
logger.info('[AI-MAP]-prdt_feas_col_path: %s' % prdt_feas_col_path)
logger.info('[AI-MAP]-client_feas_col_number_path: %s' % client_feas_col_number_path)
logger.info('[AI-MAP]-mean_std_path: %s' % mean_std_path)
logger.info('[AI-MAP]-discrete_path: %s' % discrete_path)
logger.info('[AI-MAP]-quantile_path: %s' % quantile_path)
logger.info('[AI-MAP]-train_flag: %s' % train_flag)
logger.info('[AI-MAP]-number_flag: %s' % number_flag)
logger.info('[AI-MAP]-result_train_sample_path: %s' % result_train_sample_path)
logger.info('[AI-MAP]-kerberos_file: %s' % keytab_file)

# local_file   类型：string   默认值：无  说明： 训练数据拉取到本地的地址
# hdfs_path   类型：string   默认值：None  说明： 训练数据在hdfs上面的地址
# user_cols_file   类型： string  默认值：无  说明： 客户特征文件
# prdt_cols_file   类型：string   默认值：无  说明： 产品特征文件
# user_number_feas_file   类型：string   默认值：无  说明：客户数值型特征文件
# mean_std_path  类型： string  默认值：无  说明：mean_std文件
# discrete_path  类型： string  默认值：无  说明：discrete文件
# train_flag 类型： string  默认值：1  说明：数据的格式
# number_flag  类型： string  默认值：1  说明：数值类型的特征处理方式
# result_train_sample_path 结果文件存放地址
# keytab_file  类型：string 默认值：无 说明：kerberos认证用户的秘钥

user_columns = pd.read_csv(client_feas_col_path, header=None, sep="\t")
user_columns = np.array(user_columns).tolist()[0]
print('user_columns', len(user_columns))

fund_columns = pd.read_csv(prdt_feas_col_path, header=None, sep="\t")
fund_columns = np.array(fund_columns).tolist()[0]
print('fund_columns', len(fund_columns))

user_and_fund_columns = user_columns + fund_columns
all_total_columns = ['label']
all_total_columns.extend(user_and_fund_columns)

number_columns = pd.read_csv(client_feas_col_number_path, header=None)
number_columns = np.array(number_columns).tolist()
number_columns = [item[0] for item in number_columns]
# fund number columns
number_columns.extend(['nav_total', 'issuevol'])
print('-----------number columns-------')
print(number_columns)
print(len(number_columns))

# 此函数没有用过
def birth_handle(item):
    if item is not None and item != 'None' and len(item.strip()) == 4:
        return item.strip()
    else:
        result = 1960 if item is None or item == 'None' else int(item) / 10000
        return str(int(result))


def none2zero(item):
    return '0' if item is None or item == 'None' else item.strip()


def get_data(col_names):
    """
    Get data from local_file,if hdfs_path is not None, get data from HDFS file.
    :param col_names:
    :return:
    """
    try:
        logger.info("[AI-MAP]- read data and do some transformation")
        # train_sample_path
        data = pd.read_csv(train_sample_path, header=None, names=col_names, sep='\t', dtype='str')
        print(data.shape)
        print(data.head(5))
        print('-------------col names------------')
        print(col_names)
        print(len(col_names))

        # transform the None to 0, and transform the data type
        count = 0
        tot_count = len(col_names)
        for column in data.columns:
            count += 1
            print('%s : %d - %d' % (column, count, tot_count))
            data[column] = data[column].apply(none2zero)
            if column in number_columns:
                data[column] = data[column].astype('float32')

        return data
    except:
        raise Exception("[ERROR]-[AI-MAP]: GenDfmSample Module Execute Failed！")


def __do_discrete(data, values):
    """
    do discretion
    :param data:
    :param values:
    :return:
    """
    try:
        # 异常值处理为0
        if values:
            logger.info("[AI-MAP]- handle the exception data")
            for key in values.keys():
                data[key] = data[key].apply(lambda x: x if x in values[key] else '0')
        else:
            logger.info("[AI-MAP]- Non-numerical feature do not exist")

        logger.info("[AI-MAP]- feature one-hot")
        x_field = dict()
        field = 0
        for column in user_and_fund_columns:
            print('col = %s' % column)
            # 数值型，保存下标索引和值
            if column in number_columns:
                if 2 == number_flag:
                    print('quantile_trans start')
                    with codecs.open(quantile_path, 'r', encoding="utf-8") as f:
                        quantile = json.load(f)
                    data[column] = data[column].apply(__num_dis, args=[quantile[column],])

                    data[[column + str(i) for i in pd.get_dummies(data[column]).columns]] = pd.get_dummies(data[column])
                    for i in pd.get_dummies(data[column]).columns:
                        x_field[column + str(i)] = field
                    data = data.drop(column, axis=1)  # 删除列
                else:
                    x_field[column] = field
            else:
                # 枚举型，one-hot编码
                data[[column + str(i) for i in pd.get_dummies(data[column]).columns]] = pd.get_dummies(
                    data[column])
                for i in pd.get_dummies(data[column]).columns:
                    x_field[column + str(i)] = field
                data = data.drop(column, axis=1)  # 删除列
            field += 1
        numeric_cols = list(set(data.columns) - set(['client_id', 'prdt_code']))
        #
        # data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, downcast="unsigned")
        data[numeric_cols] = data[numeric_cols].astype('float32')
        return x_field, data

    except:
        raise Exception("[ERROR]-[AI-MAP]: GenDfmSampleNormal Module Execute Failed！")


def gen_ffm_sample():
    """
    generate the DFM train or test samples
    :return:
    """
    try:
        if train_flag == 1:
            columns = all_total_columns
        else:
            columns = ['client_id', 'prdt_code']
            columns.extend(all_total_columns)

        data = get_data(columns)

        # z-score normalization
        if 1 == number_flag:
            print('mean_std_trans start')
            with codecs.open(mean_std_path, 'r', encoding="utf-8") as f:
                mean_std = json.load(f)
            if mean_std:
                for c in number_columns:
                    # 将数值类型的数归一化
                    data[c] = (data[c] - mean_std[c][0]) / mean_std[c][1]
            else:
                logger.info("[AI-MAP]- mean_std not exist")

        print("one-hot encoding start")
        with codecs.open(discrete_path, 'r', encoding="utf-8") as f:
            values = json.load(f)
        x_field, data = __do_discrete(data, values)

        logger.info("[AI-MAP]- save pkl file")
        # result_file = "dfm_train_sample"
        result_file_name = train_sample_path.split('/')[-1].replace('sample', 'dfm_sample')
        f = open(result_file_name, 'wb')
        pk.dump((x_field, data), f, 2)
        f.close()

        with open(result_flag, 'w') as f_r:
            f_r.write('True')
        f_r.close()

        # 上传到指定hdfs地址
        command = 'export HADOOP_USER_NAME=u006586;hadoop fs -rm %s' % result_train_sample_path + result_file_name
        print(command)
        os.system(command)
        command = 'export HADOOP_USER_NAME=u006586;hadoop fs -put %s %s' % (result_file_name, result_train_sample_path)
        print(command)
        cmd_status = os.system(command)
        if cmd_status != 0:
            raise Exception("COMMAND : %s " % command, "FAILED!!!")
    except:
        raise Exception("[ERROR]-[AI-MAP]: GenDfmSampleNormal Module Execute Failed！")


def __num_dis(item, *args):
    """
    :param item: data
    :param args: The quartile of a column of data
    :return: pos: The result of data discretization
    """
    quantile_col = args[0]
    if item > quantile_col[2]:
        return 3
    elif item > quantile_col[1]:
        return 2
    elif item > quantile_col[0]:
        return 1
    else:
        return 0

if __name__ == '__main__':
    logger.info("[AI-MAP]-START GenFFMSample Module!")
    print(os.listdir('.'))
    # kerberos authentication
    # keytab = keytab_file.strip('\n').strip().split('/')[-1]
    user = keytab_file.strip('\n').strip().split('/')[-1].split('.')[0]
    command = "kinit -kt %s %s" % (keytab_file, user)
    print(command)
    cmd_status = os.system(command)
    if cmd_status != 0:
        raise Exception("COMMAND : %s " % command, "FAILED!!!")
    gen_ffm_sample()
    logger.info("[AI-MAP]-END GenFFMSample Module!")