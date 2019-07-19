# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:38:04 2018

@author: 006702
"""
import pandas as pd
import numpy as np
from util import GlobalList as gl

def groupReturnCal(stock_info,pct_chg_data,group_return_pre):
    if(0==len(stock_info) or (0==len(pct_chg_data))):
        gl.logger.error('The input is empty,plz check!')
    stock_list=stock_info.iloc[:,0]
    stock_weight=stock_info.iloc[:,1]
    stock_inter=np.intersect1d(stock_list,pct_chg_data.iloc[:,1])
    stock_index=np.array([np.where(pct_chg_data.iloc[:,1]==stock_inter[i])[0][0] for i in range(len(stock_inter))]).reshape([len(stock_inter),])
    stock_pct_chg=pct_chg_data.iloc[stock_index,:]
    stock_pct_chg.index=range(len(stock_pct_chg))
    #权重所在的位置
    index_flag=np.in1d(stock_list,stock_inter)
    stock_inter_weight=stock_weight[index_flag]
    group_return=pd.Series()
    group_return['data_date']=pct_chg_data.iloc[0,0]
    group_return['group_return']=np.dot(stock_inter_weight.T,stock_pct_chg.iloc[:,2])/100
    if len(group_return_pre)==0:
        group_return['group_acc_return']=1
    elif group_return_pre['group_acc_return'].values>0:
        group_return['group_acc_return']=(group_return['group_return']+1)*group_return_pre.iloc[0,1]
    else:
        gl.logger.error('The net values is less than zero,plz check!')
    return group_return                   
