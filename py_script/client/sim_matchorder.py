# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

wait_time = 20
def read_tick(code, date):
    # 以gb2312编码方式读取xxx.csv
    df = pd.read_csv('/app/htdata/alldata/origin/code_date/%s/%s/%s_%s.csv'%(code, date, date, code), encoding='gb2312')
    return df

# 匹配订单系统函数，传入股票代码，买入时间，价格，数量
def match_order(code, date, buy_time, buy_price, buy_vol):  
    tick_data = read_tick(code, date)       # 从真实数据中读取到tick_data
    act_price = 0
    act_vol = 0
    ahead_vol = 0 # 这是啥？
    # 取到MDTime==buy_time的index的数值
    index = tick_data[(tick_data.MDTime==buy_time)].index.tolist()[0]
    tick_index = 0
    i = index + 1           # 为啥行号+1
    total_index = index + wait_time      # 如果行号+等待时间(20)比最后一行大,那么就把最后一行的行号给total_index
    if total_index > tick_data.index[-1]:
        total_index = tick_data.index[-1]
    
    while i <= total_index:  # 只有当进入最后20行的时候,i>total_index,才会跳出循环（除break）
        next_lastpx = tick_data.get_value(i, 'LastPx') #下一个最新价
        pre_vol = tick_data.get_value(i - 1, 'TotalVolumeTrade') #当前成交总量
        next_vol = tick_data.get_value(i, 'TotalVolumeTrade') #下一个成交总量
        buy1vol = tick_data.get_value(i - 1, 'Buy1OrderQty') #当前买1量
        next_buy1vol = tick_data.get_value(i, 'Buy1OrderQty') #下一个买1量
        buy1price = tick_data.get_value(i - 1, 'Buy1Price') #当前买一价
        next_buy1price = tick_data.get_value(i, 'Buy1Price') #下一个买1价
        
        if buy_price == buy1price:
            if i-1 == index:
                ahead_vol = buy1vol - buy_vol  # 这个是干什么的？
        #print(ahead_vol)
        
        if next_lastpx < buy_price:
            act_price = next_lastpx
            act_vol = buy_vol
            tick_index = i
            #print(1)
            break
        elif next_lastpx > buy_price:
            act_price = 0
            act_vol = 0      
            if next_buy1price == buy1price and buy1price == buy_price:
                # 怎么可能能执行到下一句？
                if next_buy1vol < buy1vol:
                    ahead_vol = ahead_vol - (buy1vol - next_buy1vol)
            #print(2)
        else:
            if next_vol == pre_vol:
                act_price = 0
                act_vol = 0
                if next_buy1price == buy1price and buy1price == buy_price:
                    if next_buy1vol < buy1vol:
                        ahead_vol = ahead_vol - (buy1vol - next_buy1vol)
                #print(3)
            elif next_vol > pre_vol: # 
                change = next_vol - pre_vol
                #print(next_vol, pre_vol)
                if change <= ahead_vol:
                    act_price = 0
                    act_vol = 0
                    ahead_vol = ahead_vol - change
                    #print(4)
                else:
                    if change - ahead_vol >= buy_vol:
                        act_price = buy_price
                        act_vol = buy_vol
                        tick_index = i
                        #print(5)
                        break
                    else:
                        act_price = buy_price
                        act_vol = change - ahead_vol
                        buy_vol = buy_vol - act_vol
                        tick_index = i
                        #print(6)
                        break
        i = i + 1
    if act_vol == 0:
        tick_index = i - 1
    trade_tick = tick_data.get_value(tick_index, 'MDTime')
    return act_price, act_vol, trade_tick

'''
if __name__ == '__main__':
    code = '300699'
    date = '20180619'
    buy_time = 101012000
    buy_price = 45.48
    buy_vol = 500
    act_price, act_vol, trade_tick = match_order(code, date, buy_time, buy_price, buy_vol)
    print(act_price, act_vol, trade_tick)
'''              
                
                
                
                
                
                
                
                
                
                
                
