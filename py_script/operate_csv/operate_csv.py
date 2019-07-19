# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
df = pd.read_csv('603898.csv', encoding='gb2312')
df.rename(columns={'时间':'MDTime', '最新价':'LastPx', '累计成交量':'TotalVolumeTrade', '累计成交额':'TotalValueTrade'}, inplace = True)
df.rename(columns={'最高价':'HighPx', '最低价':'LowPx', '开盘价':'OpenPx', '昨收盘':'PreClosePx'}, inplace = True)
df.rename(columns={'卖10价':'Sell10Price', '卖9价':'Sell9Price', '卖8价':'Sell8Price', '卖7价':'Sell7Price', '卖6价':'Sell6Price'}, inplace = True)
df.rename(columns={'卖5价':'Sell5Price', '卖4价':'Sell4Price', '卖3价':'Sell3Price', '卖2价':'Sell2Price', '卖1价':'Sell1Price'}, inplace = True)
df.rename(columns={'买1价':'Buy1Price', '买2价':'Buy2Price', '买3价':'Buy3Price', '买4价':'Buy4Price', '买5价':'Buy5Price'}, inplace = True)
df.rename(columns={'买6价':'Buy6Price', '买7价':'Buy7Price', '买8价':'Buy8Price', '买9价':'Buy9Price', '买10价':'Buy10Price'}, inplace = True)
df.rename(columns={'卖10量':'Sell10OrderQty', '卖9量':'Sell9OrderQty', '卖8量':'Sell8OrderQty', '卖7量':'Sell7OrderQty', '卖6量':'Sell6OrderQty'}, inplace = True)
df.rename(columns={'卖5量':'Sell5OrderQty', '卖4量':'Sell4OrderQty', '卖3量':'Sell3OrderQty', '卖2量':'Sell2OrderQty', '卖1量':'Sell1OrderQty'}, inplace = True)
df.rename(columns={'买1量':'Buy1OrderQty', '买2量':'Buy2OrderQty', '买3量':'Buy3OrderQty', '买4量':'Buy4OrderQty', '买5量':'Buy5OrderQty'}, inplace = True)
df.rename(columns={'买6量':'Buy6OrderQty', '买7量':'Buy7OrderQty', '买8量':'Buy8OrderQty', '买9量':'Buy9OrderQty', '买10量':'Buy10OrderQty'}, inplace = True)
df.drop(['万得代码', '名称', '日期', '成交量', '成交额', '成交笔数', 'IOPV/利息', 
        '成交标志', '买卖标志', '加权平均叫卖价', '加权平均叫买价', '叫卖总量', 
        '叫买总量'],axis=1, inplace=True)
df.drop([0], inplace=True)
for i in range(df.shape[0]):
    df.iat[i,0] = int(df.iat[i,0] )
df_am = df[np.array(df['MDTime']>=93000000) & np.array(df['MDTime']<113000000)]
df_pm = df[np.array(df['MDTime']>=130000000) & np.array(df['MDTime']<150000000)]
result = pd.concat([df_am, df_pm])
result.to_csv('result.csv')