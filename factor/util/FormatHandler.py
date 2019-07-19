# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:29:21 2018

@author: K0570052
"""
import numpy as np
import pandas as pd
from dao import OracleManager as om

# edit by Amoxz
def transform_large_df(oracle_data):
    df = oracle_data.pivot(index=oracle_data.columns[1], 
                            columns=oracle_data.columns[0], values=oracle_data.columns[2])
    df = df.sort_index(ascending=False)
    return df

# 处理回归问题,和别的区别在于 index 和columns正好相反
def transform_preclose(oracle_data):
    df = oracle_data.pivot(index=oracle_data.columns[0],
                            columns=oracle_data.columns[1], values=oracle_data.columns[2])
    df = df.sort_index(ascending=False)
    return df

# factor_df = factor_df.where(factor_df.notna(), None)
# factor_df = factor_df.loc[stock_list_global]
def df_to_list(cur_date, factor_df, factor_name, stock_list):
    factor_df = factor_df.loc[stock_list]
    factor_df = pd.concat([pd.DataFrame(columns=['a', 'b', 'c']), factor_df])
    factor_df['a'] = cur_date
    factor_df['b'] = factor_df.index
    factor_df['c'] = factor_name
    factor_df = factor_df.where(factor_df.notna(), None)
    factor_list = factor_df.values.tolist()

    # factor_df['flag'] = [1 if i[0] in ['0', '3', '6'] else 0 for i in factor_df.index]
    # factor_df = factor_df[factor_df.iloc[:, -2] == 1]
    # factor_df = factor_df.drop('flag', axis=1)
    #
    # factor_df = factor_df.fillna('')
    # factor_list = factor_df.values.tolist()
    # for line in range(len(factor_list)):
    #     if factor_list[line][3] == '':
    #         factor_list[line][3] = None
    
    factor_flag = om.factor_insert(factor_list)
    return factor_flag


def transform_data(oracle_data):
    data_code = oracle_data[0]  
    data_date = oracle_data[1] 
    factor_date = sorted(data_date.unique()) 
    stock_list =  sorted(data_code.unique())     
    factor_data = pd.DataFrame(index = factor_date, columns = stock_list)
    # sort = oracle_data.sort_index()
    for n in oracle_data.index:
        factor_data.at[oracle_data.at[n,1],oracle_data.at[n,0]]=oracle_data.at[n,2]
    return factor_data


def oracel_format(data,date,factor_name):
    df=pd.DataFrame(index=range(len(data)),columns=['data_date','stock_code','factor_name','factor_value'])
    for i in range(len(data)):
        df['data_date'][i]=date
        df['stock_code'][i]=data.index[i]
        df['factor_name'][i]=str(factor_name)
        df['factor_value'][i]=data[i]
    return df


def median_filter(factor_df,mad=3,flag=1): 
    if flag:
        factor_dict=factor_df.copy()
        factor_dict=factor_dict.fillna(0).iloc[:3650] # 后面的都是A开头的~
        factor_dict.index = factor_dict.iloc[:,0]
        factor_dict.drop(factor_dict.columns[0], axis=1, inplace=True) # 删除列
        date_num,stock_num = factor_dict.shape
        dm = factor_dict.median(axis=0)
        #    
        format_dm=pd.DataFrame(np.tile(dm.T,(date_num,1)))
        format_dm.index=factor_dict.index
        format_dm.columns=factor_dict.columns
        #    
        dm1=(abs(factor_dict-format_dm)).median(axis=0)
        fac_ub = pd.DataFrame(np.tile((dm + mad * 1.483 * dm1).T,[date_num,1]),\
                              index=factor_dict.index,columns=factor_dict.columns)
        fac_lb = pd.DataFrame(np.tile((dm - mad * 1.483 * dm1).T,[date_num,1]),\
                              index=factor_dict.index,columns=factor_dict.columns)
        
        factor_dict[factor_dict>fac_ub] = fac_ub
        factor_dict[factor_dict<fac_lb] = fac_lb
    else:
        factor_dict=factor_df       
    return factor_dict


def norm(factor_dict,flag=True):
    if flag:
        factor_dict[factor_dict>10**10]=10**10
        factor_dict[factor_dict<-10**10]=-10**10
        norm_factor=(factor_dict-factor_dict.mean(axis=0))/factor_dict.std()
    else :
        norm_factor=factor_dict
    return norm_factor