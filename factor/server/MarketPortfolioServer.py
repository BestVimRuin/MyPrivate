# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:09:08 2018

@author: 006702
"""
import datetime
import sys
import pandas as pd
import numpy as np 
import time
from sklearn import preprocessing
# from quantAPI import tradingDay
from calculation import FactorPrepare
from dao import OracleManager as om
# from dao import heatCountFromOracle as hm
from util import DataProcess as dp
from calculation import ModelXgboost as mxgb
from util import GlobalList as gl

   
def dataPrepare(date_list,T,N):
    if(len(date_list)!=T+N+1):
        gl.logger.error('Date list length is incorrect,plz check!')
    else:
        gl.logger.info('Prepare close data and factor data.')
        close_data=om.getCloseData(date_list[-1],date_list[0])
        train_factor_data=om.getFactorData(date_list[-1],date_list[-T])
        gl.logger.info('Close data length: %d, factor data length: %d' %(len(close_data),len(train_factor_data)))
        
        while(0==len(close_data) or 0==len(train_factor_data)):
            gl.logger.warning('Failed to get the daily data,sleep to try again!')
            time.sleep(100)
            close_data=om.getCloseData(date_list[-1],date_list[0])
            train_factor_data=om.getFactorData(date_list[-1],date_list[-T])
            
        #测试样本构建
        gl.logger.info('Prepare train data...')

        close_df = dp.closeToDf(close_data)
        gl.logger.info('Close data OK!')

        return_df = dp.calculateReturn(T,close_df)
        gl.logger.info('Return data OK!')

        train_factor_df = dp.factorDataToDf(train_factor_data)
        #训练样本 ，当天的因子数据
        gl.logger.info('Prepare test data...')
        test_factor_data=om.getFactorData(date_list[0],date_list[0])
        test_factor_df = dp.factorDataToDf(test_factor_data)       
    return return_df,train_factor_df,test_factor_df



def modelPredict(train_factor_df,return_df,test_factor_df):
     #模型部分
    gl.logger.info('Start to produce model sample!')
    train_sample= dp.produceSample(train_factor_df,return_df,'train')
    test_sample= dp.produceSample(test_factor_df,pd.DataFrame(),'test')
    gl.logger.info('Train sample number: %d, test sample number: %d' %(len(train_sample),len(test_sample)))
    ypred_df=mxgb.classificationModel(train_sample,test_sample)
    
    gl.logger.info('Model task is finished ,next to add the market_value column!')
    ypred_df.index=ypred_df.iloc[:,:2]
    test_factor_df.index=test_factor_df.iloc[:,:2]
    stock_info=pd.concat([ypred_df,test_factor_df.ix[:,'ln_capital']],axis=1) #预测值+市值，便于后面计算权重
    stock_info.ix[stock_info['ln_capital']<0,'ln_capital']=1
    stock_info.index=range(len(stock_info))
    test_factor_df.index=range(len(test_factor_df))
    
    gl.logger.info('Model task is finished ,predict data length is :%d!'%(len(ypred_df)))
    flag=om.stockRatingInsert(stock_info)
    gl.logger.info('Stock rating info insert is done,flag is :%d!'%(flag))

    return flag


def factorDataNormalized(test_factor_df):
     #风格因子信息
    factor_col = []
    factor_info=om.getStyleFactorInfo()
    style_factor=factor_info[factor_info[:,1]!='S000000',:]
    factor_col=np.append(['date','stockcode'],style_factor[:,0]) 
    #风格因子值，并全局标准化
    gl.logger.info('Start to normalize the factor data.')
    factor_list = []
    factor_col_list = list(factor_col)
    for element in factor_col_list:
        factor_list.append(str(element).strip())
    test_factor_df = test_factor_df[factor_list]
    column_mean = test_factor_df.mean(axis=0)
    for column in test_factor_df.columns:
        if column != 'stockcode':
            test_factor_df[column] = test_factor_df[column].fillna(column_mean[column])
    factor_data_scaled=preprocessing.scale(test_factor_df.ix[:,2:])
    factor_data_scaled=np.where(factor_data_scaled>3,3,factor_data_scaled)
    factor_data_scaled=np.where(factor_data_scaled<-3,-3,factor_data_scaled)

    factor_data_scaled_df = pd.DataFrame(factor_data_scaled,columns=style_factor[:,0])#全市场标准化
    factor_normal_data=pd.concat([test_factor_df.ix[:,:2],factor_data_scaled_df],axis=1)
    data_df=factor_normal_data.copy()
    #转换为库中数据格式
    factor_normal_info=dp.transferToFactordata(data_df)
    factor_normal_info.iloc[:,0]=factor_normal_info.iloc[:,0].astype('int32')
    gl.logger.info('Factor data normalization is done, stock num is :%d ,factor num is %d, record num is: %d'%(len(factor_data_scaled),len(factor_data_scaled_df.columns),len(factor_normal_info)))
    #插入因子标准数据表
    flag=om.normalDataInsert(factor_normal_info)
    gl.logger.info('Factor normalized data insert is done,flag is :%d!'%(flag))
    return flag


def  dailyCalculation(cur_date,T,N):
    gl.logger.info('Start to do daily caculation!')
    #日期获取
    date_list=om.getNtradeDate(cur_date,-(T+N+1))
    gl.logger.info('Date list get successfully,date interval is [%d,%d]'%(date_list[0],date_list[-1]))
   
    #收盘价数据、因子数据准备
    gl.logger.info('Start to do data prepare!')
    date_time = datetime.datetime.now()
    [return_df,train_factor_df,test_factor_df]=dataPrepare(date_list,T,N)
    cost_seconds=(datetime.datetime.now()-date_time).seconds
    gl.logger.info('Data preparation is done,consumes %d seconds.'%cost_seconds)
    
    #模型预测
    model_task_flag=modelPredict(train_factor_df,return_df,test_factor_df)
    
   #标准化的因子数据
    normalized_flag=factorDataNormalized(test_factor_df)
    
    if(model_task_flag==1 and normalized_flag==1):
        gl.logger.info('Daily stock rating task successfully! ')
    else:
        gl.logger.error('The model falg is %d, the normalized data flag is %d,daily task is failed, please check!'%(model_task_flag,normalized_flag))
        
    
#    风格因子多空收益计算
#    gl.logger.info('Start to get the factor info!')
#    style_flag=stylePerformCal(train_factor_df,return_df,date_list)
#    gl.logger.info('Style factor calculation is done!')
    
    
    
            
def dataCheck(data_date):
    if(0==data_date):
        gl.logger.error('The input is empty,plz check!')
        return 0
    update_status=om.getUpdateStatus(gl.TABLE_FACTOR_DATA,data_date)
    return update_status


if __name__ == '__main__':

    __console_stdin__ = sys.stdin
    __console_stdout__ = sys.stdout
    __console_stderr__ = sys.stderr

    stdin_handler = open(gl.workpath + 'stdin','w')
    stdout_handler = open(gl.workpath + 'stdout','w')
    stderr_handler= open(gl.workpath + 'stderr','w')
    
    sys.stdin = stdin_handler
    sys.stdout = stdout_handler
    sys.stderr = stderr_handler
    
    
    
    cur_date_time = datetime.datetime.now()
    cur_date_str = cur_date_time.strftime('%Y%m%d')
    cur_date = int(cur_date_str)
    
    cur_date = 20190116

    factor_prepare_flag = FactorPrepare.factor_prepare(cur_date)
    

#    cur_date_str = '20180608' # 手动设置日期S
    
#    if(cur_date!=tradingDay(cur_date,1)[0]):
#        gl.logger.info('Input date is not a tradingDay,date task ignore.')
#        sys.exit(0) 

#    hm.res_count(cur_date_str)
    

    T = 2
    N = 2
    gl.logger.info('Date:%d, now doing the data update check!' %cur_date)
    # status=dataCheck(cur_date)
    # while(0==status or 0==len(status) or 1!=status[0][0]):
    #     gl.logger.warning('Factor data is not the newest,sleep for another again!')
    #     print(datetime.datetime.now())
    #     time.sleep(1800)
    #     status=dataCheck(cur_date)
    dailyCalculation(cur_date,T,N)
    
    sys.stdin = __console_stdin__
    sys.stdout = __console_stdout__
    sys.stderr = __console_stderr__
    
    stdin_handler.close()
    stdout_handler.close()
    stderr_handler.close()