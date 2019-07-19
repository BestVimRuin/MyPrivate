# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:35:50 2018

@author: 006702
"""
import pandas as pd
from util import GlobalList as gl

def label_f(x,upper,lower):
    if x>=upper:
        return 1
    elif x<=lower:
        return 0
    else:
        return 100

# edit by Amoxz
def transform_large_df(close_data):
    data_code = close_data[1]  #股票代码
    data_date = close_data[0]  #数据日期
    df = close_data.pivot(index=close_data.columns[1], 
                            columns=data_code, values=close_data.columns[2])
    df = df.sort_index(ascending=False)
    return df

#close_data转为 date*stock
def closeToDf(close_data):
    close_df=pd.DataFrame()
    if len(close_data) ==0:
        gl.logger.error('close data is empty,plz check!')
        return close_df
    
    data_code = close_data[1]  #股票代码
    data_date = close_data[0]  #数据日期
     
    factor_date = sorted(data_date.unique())
    stock_list =  sorted(data_code.unique())
     
    close_df=pd.DataFrame(index = factor_date,columns = stock_list)
    for n in range(close_data.shape[0]):
        close_df.at[close_data.at[n,0],close_data.at[n,1]]=close_data.at[n,3]
    close_df = close_df.astype('float64')
    return close_df


#因子数据转成DataFrame
def factorDataToDf(factor_data):
    factor_df=pd.DataFrame()
    if len(factor_data) ==0:
        gl.logger.error('factor data is empty,plz check!')
        return factor_df
    
    data_factor = factor_data[2]  #因子ID
    factor_list = sorted(data_factor.unique())
    for i in range(len(factor_list)):
        tmp_col=list(('date','stockcode'))
        tmp_factor_data = factor_data.ix[factor_data.iloc[:,2]==factor_list[i],[0,1,3]]
        tmp_factor_data.index = tmp_factor_data.iloc[:,:2]
        tmp_col.append(factor_list[i])
        tmp_factor_data.columns=tmp_col
        if i==0:
            factor_df=tmp_factor_data
        else:
            factor_df=pd.concat([factor_df,tmp_factor_data.iloc[:,-1]],axis=1)
    df_len=len(factor_df)
    for index_num in range(len(factor_df.index)):
        factor_df.iloc[index_num, 0] = list(factor_df.index)[index_num][0]
        factor_df.iloc[index_num, 1] = list(factor_df.index)[index_num][1]
    factor_df.index=list(range(df_len))
    return factor_df


def factor2df_amoxz(factor_data):
    factor_df=pd.DataFrame()
    if len(factor_data) ==0:
        gl.logger.error('factor data is empty,plz check!')
        return factor_df
    # data_factor = factor_data[2]  #因子ID
    # factor_list = sorted(data_factor.unique())
    factor_df = factor_data.pivot(index=factor_data.iloc[:,:0], 
                            columns=factor_data.iloc[:,2], values=factor_data[3])
    factor_df = factor_df.sort_index(ascending=False)  # 降序
    return factor_df

#由close计算样本应周期收益
def calculateReturn(T,close_df):
    index_date=close_df.index
    column_stock=close_df.columns
    df_len=index_date.size*column_stock.size
    return_df=pd.DataFrame(index=range(df_len),columns=list(('date','stockcode','label')))    
    if (T<=0 or len(close_df)==0):
        gl.logger.warning('Hold period is incorrect or close df is empty,plz check!')
        return return_df    
    return_mat = close_df.shift(-T-1)/close_df.shift(-1) - 1
    for t in range(index_date.size):
        for n in range(column_stock.size):
            return_df.at[n+t*column_stock.size,'date']=index_date[t];
            return_df.at[n+t*column_stock.size,'stockcode']=column_stock[n]
            return_df.at[n+t*column_stock.size,'label']=return_mat.ix[index_date[t],column_stock[n]]
        return_df=return_df.dropna()
    return return_df




#dataframe转成因子数据:date,code,factor_id,factor_value
def transferToFactordata(data_df):    
    data_date=data_df.iloc[0,0]
    stock_code=data_df.iloc[:,1].values
    column_list=data_df.columns[2:]
    data_df.index=stock_code

    df_len=stock_code.size*column_list.size
    factor_data=pd.DataFrame(index=range(df_len),columns=(['data_date','stock_code','factor_id','factor_value']))
    for t in range(stock_code.size):
        for n in range(column_list.size):
            factor_data.at[n+t*column_list.size,'data_date']=data_date;
            factor_data.at[n+t*column_list.size,'stock_code']=stock_code[t];
            factor_data.at[n+t*column_list.size,'factor_id']=column_list[n]
            factor_data.at[n+t*column_list.size,'factor_value']=data_df.ix[stock_code[t],column_list[n]]       
    return factor_data




#构造训练样本、测试样本
#sample_flag='train' or 'test'
def produceSample(factor_df,return_df,sample_flag):
    if (len(factor_df)==0  or sample_flag not in list(('train','test'))):
        gl.logger.error('Factor df isempty,or sample flag is incorrect!')
        return pd.DataFrame()
    sample_df=factor_df.copy()
    if sample_flag=='test':
        data_date=factor_df['date'].unique()
        for t in range(len(data_date)):
            tmp_factor_df=factor_df.ix[factor_df.iloc[:,0]==data_date[t],2:]
            sample_df.ix[factor_df.iloc[:,0]==data_date[t],2:]=tmp_factor_df.rank(method='first',pct=True)
    elif  sample_flag=='train':
        sample_df.index=factor_df.iloc[:,:2]
        return_df.index=return_df.iloc[:,:2]
        factor_label_df=pd.concat([sample_df,return_df.iloc[:,-1]],join='inner',axis=1)
        data_date=factor_df['date'].unique()
        for t in range(len(data_date)):
            tmp_factor_df=factor_label_df.ix[factor_label_df.iloc[:,0]==data_date[t],2:-1]
            tmp_label_df=factor_label_df.ix[factor_label_df.iloc[:,0]==data_date[t],-1]
            factor_label_df.ix[factor_label_df.iloc[:,0]==data_date[t],2:-1]=tmp_factor_df.rank(method='first',pct=True)
            lower=tmp_label_df.quantile(q=1/5)
            upper=tmp_label_df.quantile(q=4/5)
            factor_label_df.ix[factor_label_df.iloc[:,0]==data_date[t],-1]=tmp_label_df.apply(label_f,args=(upper,lower,))
        sample_df=factor_label_df[factor_label_df.iloc[:,-1]!=100]
    return_df.index=range(len(return_df))
    sample_df.index=range(sample_df.shape[0])
    return sample_df
        
        

        
            
        
    
    



  