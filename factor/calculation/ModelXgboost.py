# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:49:39 2018

@author: 006702
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from util import GlobalList as gl


def ceateFeatureMap(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close() 
    


#分类器模型
def classificationModel(train_data,test_data):
    if(0==len(train_data) or 0==len(test_data)):
        gl.logger.error('Train data or test data is empty,plz check!')
        return pd.DataFrame()
    df_index=test_data.iloc[:,0:2]
     #    测试集分割
    gl.logger.info('Split the train & test data:')
    df_train_data_org, df_train_data_eval,df_train_label_org, df_train_label_eval =\
    train_test_split(train_data.iloc[:,2:-1],train_data.iloc[:,-1], test_size=0.2)
    #   构建xgb模型输入数据
    feature_train_org=np.array(df_train_data_org.values)
    label_train_org=np.array(df_train_label_org.values)
    dtrain_org=xgb.DMatrix(feature_train_org,label=label_train_org) 
    
    feature_train_eval=np.array(df_train_data_eval.values)
    label_train_eval=np.array(df_train_label_eval.values)
    dtrain_eval=xgb.DMatrix(feature_train_eval,label=label_train_eval)#验证集数据
#    features = [x for x in df_train_data_org.columns]
#    ceateFeatureMap(features) 
     #测试数据
    df_test_data=test_data.iloc[:,2:]
    feature_test=np.array(df_test_data.values)
    '''参数设置'booster':'gbtree'默认为gbtree方式
    'objective':'binary:logistic'   二进制的逻辑回归，输出为涨跌概率 
    'silent':1， 默认为0，打印日志信息，1为静默执行
    'eta':0.05 缩减新特征的权重，区间【0-1】
    subsample [default=1] 用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的冲整个样本集合中随机的抽取出50%的子样本建立树模型，这能够防止过拟合。取值范围为：(0,1]
    # 'verbose':False,'''

    param={'booster':'gbtree','objective':'binary:logistic','max_depth':4,'verbose':False,'silent':0,'eta':0.05,\
            'subsample':0.8,'colsample_bylevel':0.9,'gamma':0,'eval_metric': 'error'}
#    param={'booster':'gbtree','objective':'binary:logistic','max_depth':4,'eta':0.05,\
#            'subsample':0.8,'colsample_bylevel':0.8,'eval_metric': 'error'}
    param['nthread'] = 2
    plst = param.items()    
   
    evallist  = [(dtrain_org,'train'),(dtrain_eval,'eval')]#验证数据集
    #训练
    num_round = 100
    bst = xgb.train(plst, dtrain_org, num_round,evallist)
    #预测
    ddtest=xgb.DMatrix(feature_test)
    ypred = bst.predict(ddtest)
    ypred_df=pd.DataFrame()
    ypred_df['pred']=ypred
    ypred_df=pd.concat([df_index,ypred_df],axis=1)
#    importance=bst.get_score(fmap='xgb.fmap',importance_type='gain')
    return ypred_df


def selfStockFilter(self_stock_info):
    stock_filter_info = pd.DataFrame(columns=list(('stockcode','weight','ypred')))
    self_stock_info_sort = self_stock_info.sort_values(by='pred',ascending=False) 
    
    stock_len=len(self_stock_info)
    if stock_len>1000:
        stock_filter_info['stockcode']=self_stock_info_sort.iloc[:300,1]
        stock_filter_info['ypred']=self_stock_info_sort.iloc[:300,2]
        stock_filter_info['weight']=self_stock_info_sort.iloc[:300,3]/np.sum(self_stock_info_sort.iloc[:300,3])
        
    elif  stock_len>500:
        stock_filter_info['stockcode']=self_stock_info_sort.iloc[:200,1]
        stock_filter_info['ypred']=self_stock_info_sort.iloc[:200,2]
        stock_filter_info['weight']=self_stock_info_sort.iloc[:200,3]/np.sum(self_stock_info_sort.iloc[:200,3])

        
    elif  stock_len>100:
        stock_filter_info['stockcode']=self_stock_info_sort.iloc[:100,1]
        stock_filter_info['ypred']=self_stock_info_sort.iloc[:100,2]
        stock_filter_info['weight']=self_stock_info_sort.iloc[:100,3]/np.sum(self_stock_info_sort.iloc[:100,3])

    elif  stock_len>0:
        stock_filter_info['stockcode']=self_stock_info_sort.iloc[:stock_len,1]
        stock_filter_info['ypred']=self_stock_info_sort.iloc[:stock_len,2]
        stock_filter_info['weight']=self_stock_info_sort.iloc[:stock_len,3]/np.sum(self_stock_info_sort.iloc[:stock_len,3])

    else:
        gl.logger.warning('The input is empty,plz check!')
    return stock_filter_info
    