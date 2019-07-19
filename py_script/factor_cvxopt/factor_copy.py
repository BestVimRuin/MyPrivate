# -*- coding: utf-8 -*-
import os
import logging
import json
import time
import numpy as np 
import pandas as pd
from cvxopt import solvers, matrix

solvers.options['show_progress'] = False

log_path = r'D:\gitee\htsc\factor_cvxopt\factor_cvxopt.log'
logging.basicConfig(filename='%s' % log_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def factor_cvxopt(train, num, train_num=0):
    """
    :param : df,y(产品)为第一列,其余列为x(基准)
    :return: W权重系数, r_square拟合度, w_sum权重之和, result_test预测值, r_new另一种方法的拟合度
    """

    train.iloc[:, 1:] = train.iloc[:, 1:] / train.iloc[-1, 1:]
    
    # 相除为inf的填充为0
    train = train[~train.isin([np.inf])].fillna(0)
    
    if num != -1 and num != 0:
        train = train.iloc[train_num:num+train_num, ]
    # 处理传过来的数据,变成df
    # logger.info('the func factor_cvxopt is running')

    train_index_list = list(train.index)
    logger.info('{},   {}'.format(train_index_list[0], train_index_list[-1]))
    logger.info('{}'.format(len(train_index_list)))
    
    
    y_df = train.iloc[:, 0:1]
    x_df = train.iloc[:, 1:]

    x_row, x_col = x_df.shape
    y_mean = y_df.mean(axis=0)    
    row_one = pd.DataFrame(np.ones([x_row, 1]))
    row_one.index = x_df.index
    
    x_matrix = np.matrix(x_df)
    y_matrix = np.matrix(y_df)   
    
    P = matrix(np.dot(2 * x_matrix.T, x_matrix))
    q = matrix(np.dot(-2 * x_matrix.T, y_matrix))
    
    g_zero = np.zeros([x_col*2, x_col])
    # 主要为了生成 x1<1 x2<2 .....  -x1<0  -x2<0 的np.array
    for i in range(x_col):
        g_zero[i][i] = 1
        g_zero[x_col+i][i] = -1   
    G = matrix(g_zero)
    
    h_para = np.zeros([x_col*2])
    h_para[:x_col] = 1
    h = matrix(h_para) 

    a_para = np.ones([1, x_col])
    A = matrix(a_para)

    b_para = np.ones([1, 1])
    b = matrix(b_para)

    # logger.info('prepare to starting train with solvers.qp')
    sol = solvers.qp(P, q, G, h, A, b)
    # logger.info('the func solvers.qp is successful')
    try:
        W = []
        for w_num in range(len(sol['x'])):
            W.append(sol['x'][w_num])
        # logger.info('#####################成功############################{}'.format(W))
    except Exception as err:
        logger.error('#####################错误############################')
        W = np.array([sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], ])

    # 对W求和,验证是否等于1
    w_sum = sum(W)
    # 预测的Y值
    result_test = np.dot(x_matrix, W)
    # 预测值和实际值的误差
    C = result_test - y_matrix.T
    # 残差平方和
    MSE = np.dot(C, C.T)
    # 取[]中的第0个元素
    SSE = MSE[0]
    SST = 0
    y_mean = np.tile(y_mean, (y_matrix.shape[0],1))
    SST = (np.array(y_matrix - y_mean)**2).sum()
    # 拟合度公式为 r  = (总方差-残差)/总方差 = 1 - (残差/总方差)
    r_square = 1 - float(SSE / SST)

    return W, r_square, w_sum, result_test
    

def run(message):
    train = json.loads(message)
    logger.info('acceptting the message successful')
    x_dict = train['bench']
    y_dict = train['product']
    try:
        num = train['num']
    except KeyError as err:
        num = 1400
        logger.info('num is not exist, {}'.format(err))
    result = {}
    date_dict = {}
    w_dict = {}
    for y_key in y_dict.keys():
        logger.info('{} start'.format(y_key,))
        w_dict = {}
        date_dict = {}
        x_init = pd.DataFrame.from_dict(x_dict, orient='columns')
        logger.info('train_x length is {}'.format(len(x_init)))
        y_init = pd.DataFrame.from_dict(y_dict, orient='columns')
        logger.info('train_y length is {}'.format(len(y_init)))
        train = pd.concat([y_init[y_key], x_init], axis=1, join='inner')
        logger.info('train_all length is {}'.format(len(train)))
        
        
        # 只去除 Y列为nan的
        train.dropna(subset=[y_key], how='any', inplace=True)
        train = train.fillna(0)
        
        train.sort_index(ascending=False, inplace=True)
        # 小于很小的值,报错
        if train.shape[0] < 10:
            logger.error('ERROR:The rows of  product {} is {}'.format(y_key, train.shape[0]))
            result[y_key] = {'ERROR':'100001'}
            continue
        # 小于给定的参数的值,报错
        if num != -1:
            if train.shape[0] < num:
                logger.error('ERROR:The rows of  product {} is {}'.format(y_key, train.shape[0]))
                result[y_key] = {'ERROR':'100002'}
                continue
        if num == -1:
            logger.info('num == -1 start')
            W, r_square, w_sum, result_test = factor_cvxopt(train, num)
            for num_col in range(x_init.shape[1]):
                w_dict[x_init.columns[num_col]] = list(W)[num_col]
            date_dict[list(train.index)[0]] = w_dict
        else:
            logger.info('num != -1 start')
            for train_num in range(len(train) - num):
                w_dict = {}
                W, r_square, w_sum, result_test = factor_cvxopt(train, num, train_num)
                # 把权重的值分别赋值给对应的基准
                for num_col in range(x_init.shape[1]):
                    w_dict[x_init.columns[num_col]] = list(W)[num_col]
                date_dict[list(train.index)[train_num]] = w_dict
        result[y_key] = date_dict
    logger.info('the result done')
    return result



def creat_json3():
    with open('jsonstr.txt', mode='r', encoding='utf-8') as f_product:
        my_json = f_product.read()
    return my_json

if __name__ == '__main__':
    st = time.time()
    message = creat_json3()
    result = run(message)
    # result_json = json.dumps(result)


    # 大约耗时0.04s.可以500组一次
    #for i in range(1000):
    # 分别为权重,权重之和,预测值
    # W, r_square, w_sum, result_test = factor_cvxopt()







