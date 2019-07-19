# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:43:06 2018

@author: 006702
"""

import pandas as pd 
import numpy as np
import util.GlobalList as gl
from sklearn import preprocessing
from util import GlobalList as gl



def stylePerformCal(style_factor,style_factor_df,style_return_df):
    if(0==len(style_factor) or 0==len(style_factor_df) or 0==len(style_return_df)):
        gl.logger.error('The input is empty,plz check!')
    style_factor_df.index=style_factor_df.iloc[:,:2]
    style_return_df.index=style_return_df.iloc[:,:2]
    style_data=pd.concat([style_factor_df,style_return_df['label']],join='inner',axis=1)
    style_data.index=range(len(style_data))
    style_list=np.unique(style_factor[:,1])
    style_factor_return=pd.Series()
    for style_id in style_list:
        gl.logger.info('Now handling the style calculation:'+style_id)
        factor_id=style_factor[style_factor[:,1]==style_id,0]
        tmp_col=np.append(['date','stockcode'],np.append(factor_id,['label']))
        tmp_style_data=style_data[tmp_col]
        tmp_factor_return=factorPerformCalcu(style_id, tmp_style_data)
        style_factor_return=pd.concat([style_factor_return,tmp_factor_return])
    return style_factor_return
        
        
def factorPerformCalcu(style_id,style_data):
    if(0==len(style_id) or 0==len(style_data)):
        gl.logger.error('The input is empty,plz check!')
    style_factor_id=style_data.columns
    style_factor_id=list(style_factor_id[2:-1])
    factor_return=pd.Series(index=style_factor_id)
    num=5
    for factor_id in style_factor_id:
        tmp_col=np.append(factor_id,['label'])
        factor_data=style_data[tmp_col].dropna().sort_values(by=factor_id)
        #因子值nan值过多
        if len(factor_data)<200:
            gl.logger.warning('Stock number is too small！')
            factor_return[factor_id]=0
            continue;
        block_size=int(len(factor_data)/num)
        top_return=np.mean(factor_data.iloc[:block_size,1])
        bottom_return=np.mean(factor_data.iloc[-block_size:,1])
        if style_id in list((gl.MOMENTUM_ID,gl.GROWTH_ID,gl.VOLATILITY_ID,gl.HEAT_ID)):
            factor_return[factor_id]=bottom_return-top_return
        else:
            factor_return[factor_id]=top_return-bottom_return
    tmp_factor_return=factor_return.dropna()
    factor_return[style_id]=np.mean(tmp_factor_return)
    return factor_return
        
        
        
def styleExposureCal(style_factor,stock_list,stock_weight,test_factor_df):
    if(0==len(style_factor) or 0==len(stock_list) or 0==len(test_factor_df)):
        gl.logger.error('The input is empty,plz check!')
    style_list=np.unique(style_factor[:,1])
    factor_col=np.append(['date','stockcode'],style_factor[:,0]) 
    test_factor_df=test_factor_df[factor_col]
    factor_data_scaled = preprocessing.scale(test_factor_df.ix[:,2:])#全市场标准化
    #自选股在所有股票中位置
    stock_inter=np.intersect1d(stock_list,test_factor_df.iloc[:,1])
    stock_index=np.array([np.where(test_factor_df.iloc[:,1]==stock_inter[i])[0][0] for i in range(len(stock_inter))]).reshape([len(stock_inter),])
#    stock_index=np.in1d(test_factor_df.iloc[:,1],stock_inter)
    self_data_scaled = factor_data_scaled[stock_index,:]
    #权重所在的位置
    index_flag=np.in1d(stock_list,stock_inter)
    stock_inter_weight=stock_weight[index_flag]
    #自选股权重，默认等权
    factor_exposure=np.dot(stock_inter_weight.T,self_data_scaled)
    style_exposure_df=pd.Series(factor_exposure,index=test_factor_df.columns[2:])
    for style_id in style_list:
        factor_id=style_factor[style_factor[:,1]==style_id,0]
        factor_exposure_df=style_exposure_df[factor_id].fillna(value=0)
        style_exposure_df[style_id]=np.mean(factor_exposure_df)
    return style_exposure_df
        
        


