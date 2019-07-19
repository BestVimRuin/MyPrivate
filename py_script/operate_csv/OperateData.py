# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import logging

 
# 处理一个文件夹下的数据,如需处理一个,单独执行即可
def operate_data(each_dir):
    filename = "data\\{}\\{}.csv".format(each_dir, stock_code)
    result_name =  "data\\{}\\{}_{}.csv".format(each_dir, each_dir, stock_code)
    df = pd.read_csv(filename, encoding='gb2312')
    df.rename(columns={'时间':'MDTime','成交量':'PreVolume', '最新价':'LastPx', '累计成交量':'TotalVolumeTrade', '累计成交额':'TotalValueTrade'}, inplace = True)
    df.rename(columns={'最高价':'HighPx', '最低价':'LowPx', '开盘价':'OpenPx', '昨收盘':'PreClosePx'}, inplace = True)
    df.rename(columns={'卖10价':'Sell10Price', '卖9价':'Sell9Price', '卖8价':'Sell8Price', '卖7价':'Sell7Price', '卖6价':'Sell6Price'}, inplace = True)
    df.rename(columns={'卖5价':'Sell5Price', '卖4价':'Sell4Price', '卖3价':'Sell3Price', '卖2价':'Sell2Price', '卖1价':'Sell1Price'}, inplace = True)
    df.rename(columns={'买1价':'Buy1Price', '买2价':'Buy2Price', '买3价':'Buy3Price', '买4价':'Buy4Price', '买5价':'Buy5Price'}, inplace = True)
    df.rename(columns={'买6价':'Buy6Price', '买7价':'Buy7Price', '买8价':'Buy8Price', '买9价':'Buy9Price', '买10价':'Buy10Price'}, inplace = True)
    df.rename(columns={'卖10量':'Sell10OrderQty', '卖9量':'Sell9OrderQty', '卖8量':'Sell8OrderQty', '卖7量':'Sell7OrderQty', '卖6量':'Sell6OrderQty'}, inplace = True)
    df.rename(columns={'卖5量':'Sell5OrderQty', '卖4量':'Sell4OrderQty', '卖3量':'Sell3OrderQty', '卖2量':'Sell2OrderQty', '卖1量':'Sell1OrderQty'}, inplace = True)
    df.rename(columns={'买1量':'Buy1OrderQty', '买2量':'Buy2OrderQty', '买3量':'Buy3OrderQty', '买4量':'Buy4OrderQty', '买5量':'Buy5OrderQty'}, inplace = True)
    df.rename(columns={'买6量':'Buy6OrderQty', '买7量':'Buy7OrderQty', '买8量':'Buy8OrderQty', '买9量':'Buy9OrderQty', '买10量':'Buy10OrderQty'}, inplace = True)
    df.drop(['万得代码', '名称', '日期', '成交额', '成交笔数', 'IOPV/利息', 
            '成交标志', '买卖标志', '加权平均叫卖价', '加权平均叫买价', '叫卖总量', 
            '叫买总量'],axis=1, inplace=True)
    df.drop([0], inplace=True)
    # 这部分用于李聂需要加的,如非要求,需要注释
    # df = pd.concat([df, pd.DataFrame(columns=['NumTrades','DiffPx1','DiffPx2','MaxPx','MinPx','TotalBidQty', 'TotalOfferQty',
    # 'WeightedAvgBidPx','WeightedAvgOfferPx','Buy1NumOrders','Buy1NoOrders','Sell1NumOrders','Sell1NoOrders'])])
    
    # 对每列的数据进行数字化,避免出现str
    for i in range(df.shape[1]):
        df[df.columns[i]] = pd.to_numeric(df[df.columns[i]], errors='coerce')
    # 注意此处的&,只有np.array才好用,不要随便用,参考stackoverflow
    df_am = df[np.array(df['MDTime']>=93000000) & np.array(df['MDTime']<113000000)]
    df_pm = df[np.array(df['MDTime']>=130000000) & np.array(df['MDTime']<150000000)]
    result = pd.concat([df_am, df_pm])
    result = result / 10000
    
    result['MDTime'] = (result['MDTime'] * 10000)
    result['TotalValueTrade'] = (result['TotalValueTrade'] * 10000)
    result['TotalVolumeTrade'] = (result['TotalVolumeTrade'] * 10000)
    result['PreVolume'] = (result['PreVolume'] * 10000)
    for i in range(1,11):
        result['Buy{}OrderQty'.format(i)] = (result['Buy{}OrderQty'.format(i)] * 10000)
    for i in range(1,11):
        result['Sell{}OrderQty'.format(i)] = (result['Sell{}OrderQty'.format(i)] * 10000)    
    # result = pd.concat([result, pd.DataFrame(columns=['NumTrades','DiffPx1','DiffPx2','MaxPx','MinPx','TotalBidQty', 'TotalOfferQty',
    # 'WeightedAvgBidPx','WeightedAvgOfferPx','Buy1NumOrders','Buy1NoOrders','Sell1NumOrders','Sell1NoOrders','Buy1OrderDetail','Sell1OrderDetail'])])
    result.fillna(0)
    result.to_csv(result_name,index=False)


def MyTimes():
    hour = '9'
    minute = '30'
    second = '03'
    times_list = []
    times = int(hour + minute + second)
    while times < 150000000:
        if len(str(second)) == 1:
            second = '0' + str(second)
        if len(str(minute)) == 1:
            minute = '0' + str(minute)      
        times = str(hour) + str(minute) + str(second) + '000'
        times_list.append(times)
        second = int(second) + 1
        times = int(times)
        if second == 60:
            second = 0
            minute = int(minute) + 1
            if minute == 60:
                minute = 0
                hour = int(hour) + 1
    # 返回的是list,每个元素是str
    return times_list[::3] 


def getPos(preTime, tickTimeList):
    for i in range(len(tickTimeList)-1):
        if preTime == tickTimeList[i]:
            return i
        elif preTime > tickTimeList[i] :
            i = i + 1
        elif preTime < tickTimeList[i]:
            return i-1
    return i-1


def scaleData(each_dir):
    allData = pd.read_csv('data\\%s\\%s_%s.csv'%(each_dir, each_dir, stock_code),encoding='gb2312')
    tickTimeList = allData['MDTime'].tolist()
    scaleTimeList = MyTimes()
    scaleData = pd.DataFrame()
    for preTime in scaleTimeList:
        pos = getPos(int(preTime), tickTimeList)
        everyData = allData.iloc[pos:pos+1,:]
        scaleData = pd.concat([scaleData, everyData], axis=0)
        print(preTime)
    scaleData['MDTime'] = scaleTimeList
    for i in range(scaleData.shape[1]):
        scaleData[scaleData.columns[i]] = pd.to_numeric(scaleData[scaleData.columns[i]], errors='coerce')    
    df_am = scaleData[np.array(scaleData['MDTime']>=93000000) & np.array(scaleData['MDTime']<113000000)]
    df_pm = scaleData[np.array(scaleData['MDTime']>=130000000) & np.array(scaleData['MDTime']<150000000)]
    scaleData = pd.concat([df_am, df_pm])        
    scaleData.to_csv('data\\{}\\{}_{}_scale.csv'.format(each_dir ,each_dir, stock_code),index=False, encoding='gb2312')

if __name__ == '__main__':

    cur_dir = os.path.abspath('.')      # 返回绝对路径
    data_dir = cur_dir + '\\data'       # 数据的绝对路径
    dirs = os.listdir(data_dir)         # data下的每个文件夹
    stock_list = ['512160']
    # stock_code = '601766'  
    for stock_code in stock_list:
        # 处理最开始的数据
        for each_dir in dirs:
            # 第一步处理数据
            operate_data(each_dir)  
            # 第一步的结果会作为此函数的输入
            # scaleData(each_dir)



    









