# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:48:29 2018

@author: 006702
"""
import cx_Oracle
import pandas as pd
import numpy as np
from util import GlobalList as gl

# 另一个用户连接数据库,用于插入
def getConnection():
    conn = False
    try:
        conn = cx_Oracle.connect(gl.FOF_NAME, gl.FOF_PWD, gl.FOF_PORT)
    except cx_Oracle.DatabaseError as msg:  
        gl.logger.exception(msg)
        print(msg) 
    return conn

#数据查询操作
def dataSelect(sql):
    conn=getConnection()
    if False == conn:
        gl.logger.error('The oracle connection is false,plz check!')
    cur=conn.cursor()
    try:
        res = cur.execute(sql)
        result=res.fetchall()
    except Exception as e:
        result=0
        gl.logger.exception(e)
        print(e)
    finally:
        cur.close()
        conn.close()
    return result

#数据库插入
def dataInsert(sql,data):
    conn=getConnection()
    if False == conn:
        gl.logger.error('The oracle connection is false,plz check!')
    cur=conn.cursor()
    try:
        cur.executemany(sql,data)
        conn.commit()
        flag=1
    except Exception as e:
        flag=0
        gl.logger.exception(e)
        print(e)
    finally:
        cur.close()
        conn.close()
    return flag

#获取收盘价
def getCloseData(start_date,end_date):
    close_data=pd.DataFrame()
    sql_str='select *  from ' + gl.TABLE_FACTOR_DATA + ' where FACTOR_ID=\'close_bak\' and data_date<=%d and data_date>=%d' %(end_date, start_date)
    close_data=pd.DataFrame(dataSelect(sql_str))
    return close_data

#获取因子数据
def getFactorData(start_date,end_date):
    factor_data=pd.DataFrame()    
    sql_str='select *  from ' + gl.TABLE_FACTOR_DATA + '  where FACTOR_ID!=\'close\' and FACTOR_ID!=\'pct_chg_all\'  and data_date<=' +'%d' %end_date + ' and data_date>=' +'%d' %start_date
    factor_data=pd.DataFrame(dataSelect(sql_str))
    return factor_data

# 取某个日期前(负，-7)or后(正，+7)N天的交易日日期,date_list[0]为最近的一天
def getNtradeDate(start_date, n):
    date_list = np.empty(abs(n))
    # 两种格式化皆可，下面的更容易懂，且不容易出错
    if n >= 0:
        sql_str = 'select to_number(trade_days) from (select distinct trade_days from \
                      ' + gl.TABLE_CALENDAR + ' where trade_days>= ' + '%s' % start_date + ' \
                      order by trade_days) where  rownum<=' + '%s' % n + ' order by trade_days desc'
    else:
        n = abs(n)
        sql_str = 'select to_number(trade_days) from (select distinct trade_days from \
                      ' + gl.TABLE_CALENDAR + ' where trade_days<=%s   \
                      order by trade_days desc) where  rownum<=%s order by trade_days desc' % (start_date, n)
    date_list = np.array(dataSelect(sql_str)).reshape((n,))
    return date_list        # 返回N行1列的numpy.ndarray

#查询因子信息
def getStyleFactorInfo():
    sql_str ='select  distinct factor_id,style_id from ' + gl.TABLE_FACTOR_INFO 
    factor_list=np.array(dataSelect(sql_str))
    return factor_list
    
#查询自选股信息
def getSelfStockInfo(data_date):
    sql_str ='select distinct stock_code,user_id,group_id from M_SELF_STOCK  where  data_date='+'%d' %data_date 
    stock_info=pd.DataFrame(dataSelect(sql_str),columns=['stockcode','user_id','group_id'])
    return stock_info


def getGroupInfo(data_date,group_id,user_id,group_type):
    sql_str ='select distinct stock_code,stock_weight from M_GROUP_INFO WHERE data_date='+'%d' %data_date \
              +' and group_id =\''+group_id+'\''+ ' and user_id =\''+user_id+'\''+' and group_type ='+'%d' %group_type
    stock_info=pd.DataFrame(dataSelect(sql_str))
#    print(sql_str)
    return stock_info


def getGroupReturn(data_date,group_id,user_id):
    sql_str ='select distinct group_id,group_acc_return,user_id from M_GROUP_RETURN WHERE data_date='+'%d' %data_date +' and group_id =\''+group_id+'\'' + ' and user_id =\''+user_id+'\''
    group_info=pd.DataFrame(dataSelect(sql_str),columns=['group_id','group_acc_return','user_id'])
    return group_info


def getUpdateStatus(table_name,data_date):
    sql_str ='select distinct table_status from M_TABLE_STATUS  where data_date ='+'%d' %data_date +' and table_name=\''+table_name +'\''
    update_flag=dataSelect(sql_str)
    return update_flag


# def newsHeatStatusInsert(data_date):
#     sql='insert into M_TABLE_STATUS (DATA_DATE,TABLE_NAME,TABLE_STATUS) VALUES(:1,:2,:3)'
#     news_heat_status=pd.DataFrame()
#     news_heat_status['data_date']=np.tile(data_date,1)
#     news_heat_status['table_name']=np.tile('M_NEWS_HEAT',1)
#     news_heat_status['table_status']=np.tile(1,1)
#
#     news_heat_status_list=news_heat_status.values.tolist()
#     flag=dataInsert(sql,news_heat_status_list)
#     return flag


#每日模型预测结果入库
def stockRatingInsert(stock_info):
    if(0==len(stock_info)):
        gl.logger.error('The stock info  is empty,plz check')
        return
    sql='insert into ' + gl.TABLE_FACTOR_MODEL + ' (DATA_DATA,STOCK_CODE,Y_PRED,LN_CAPITAL) VALUES(:1,:2,:3,:4)'
    stock_info_df=pd.DataFrame()
    stock_info_df['data_date']=stock_info.iloc[:,0].values
    stock_info_df['stock_code']=stock_info.iloc[:,1].values
    stock_info_df['stock_rating']=stock_info.iloc[:,2].values
    stock_info_df['market_value']=stock_info.iloc[:,3].values

    stock_info_df = stock_info_df.where(stock_info_df.notna(), None)

    stock_info_list=stock_info_df.values.tolist()
    flag=dataInsert(sql,stock_info_list)                
    return flag


#标准化之后的因子数据入库
def normalDataInsert(factor_normal_info):
    if(0==len(factor_normal_info)):
        gl.logger.error('The factor data is empty,plz check!')
        return
    sql='insert into M_FACTOR_NORMAL_DATA_TEST (DATA_DATE,STOCK_CODE,FACTOR_ID,FACTOR_VALUE) VALUES (:1,:2,:3,:4)'
    factor_normal_df=pd.DataFrame()
    factor_normal_df['data_date']=factor_normal_info.iloc[:,0].values
    factor_normal_df['stock_code']=factor_normal_info.iloc[:,1].values
    factor_normal_df['factor_id']=factor_normal_info.iloc[:,2].values
    factor_normal_df['factor_value']=factor_normal_info.iloc[:,3].values

    factor_normal_df = factor_normal_df.where(factor_normal_df.notna(), None)
    factor_normal_list=factor_normal_df.values.tolist()
    flag=dataInsert(sql,factor_normal_list)
    return flag
                    
                    
# #风格因子表现入库
# def styleReturnInsert(data_date,style_factor_return):
#     if(0==len(style_factor_return)):
#         gl.logger.error('The style_factor_return input is empty,plz check!')
#     sql='insert into M_FACTOR_RETURN (DATA_DATE,FACTOR_ID,FACTOR_RETURN) VALUES(:1,:2,:3)'
#     factor_return_df=pd.DataFrame()
#     factor_return_df['data_date']=np.tile(data_date,len(style_factor_return))
#     factor_return_df['factor_id']= style_factor_return.index
#     factor_return_df['factor_return']= style_factor_return.values
#     factor_return_list=factor_return_df.values.tolist()
#     flag=dataInsert(sql,factor_return_list)
#     return flag





# #因子暴露入库
# def factorExposureInsert(data_date,group_id,factor_exposure,benchmark,user_id):
#     if( 0==len(group_id) or 0==len(factor_exposure) or 0==len(benchmark)):
#         gl.logger.error('The style_factor_return input is empty,plz check!')
#     sql='insert into M_FACTOR_EXPOSURE (DATA_DATE,FACTOR_ID,GROUP_ID,BENCHMARK,EXPOSURE_VALUE,USER_ID) VALUES(:1,:2,:3,:4,:5,:6)'
#     factor_exposure_df=pd.DataFrame()
#     factor_exposure_df['data_date']=np.tile(data_date,len(factor_exposure))
#     factor_exposure_df['factor_id']=factor_exposure.index
#     factor_exposure_df['group_id']=np.tile(group_id,len(factor_exposure))
#     factor_exposure_df['benchmark']=np.tile(benchmark,len(factor_exposure))
#     factor_exposure_df['exposure_value']=factor_exposure.values
#     factor_exposure_df['user_id']=np.tile(user_id,len(factor_exposure))
#     factor_exposure_list=factor_exposure_df.values.tolist()
#     dataInsert(sql,factor_exposure_list)

# #组合每日信息入库
# def groupInfoInsert(data_date,stock_list,stock_weight,group_id,group_type,benchmark,user_id):
#     if( 0==len(stock_list) or 0==len(stock_weight)):
#         gl.logger.error('The style_factor_return input is empty,plz check!')
#     group_info_df=pd.DataFrame()
#     group_info_df['data_date']=np.tile(data_date,len(stock_list))
#     group_info_df['stock_code']=stock_list
#     group_info_df['stock_weight']=stock_weight
#     group_info_df['group_id']=np.tile(group_id,len(stock_list))
#     group_info_df['group_type']=np.tile(group_type,len(stock_list))
#     group_info_df['benchmark']=np.tile(benchmark,len(stock_list))
#     group_info_df['user_id']=np.tile(user_id,len(stock_list))
#     sql='insert into M_GROUP_INFO (DATA_DATE,STOCK_CODE,STOCK_WEIGHT,GROUP_ID,GROUP_TYPE,BENCHMARK,USER_ID) VALUES(:1,:2,:3,:4,:5,:6,:7)'
#     group_info_list=group_info_df.values.tolist()
#     dataInsert(sql,group_info_list)


# def groupReturnInsert(group_return,group_id,user_id):
#     if(0==len(group_return) ):
#         gl.logger.error('The style_factor_return input is empty,plz check!')
#     sql='insert into M_GROUP_RETURN (DATA_DATE,GROUP_ID,GROUP_RETURN,GROUP_ACC_RETURN,USER_ID) VALUES(:1,:2,:3,:4,:5)'
#
#     group_return_df=pd.DataFrame()
#     group_return_df['data_date']=pd.Series(data=group_return['data_date'])
#     group_return_df['group_id']=group_id
#     group_return_df['group_return']=group_return['group_return']
#     group_return_df['group_acc_return']=group_return['group_acc_return']
#     group_return_df['user_id']=user_id
#     group_return_list=group_return_df.values.tolist()
#     dataInsert(sql,group_return_list)


def factor_insert(value):
    sql = 'insert into ' + gl.TABLE_FACTOR_DATA + '(DATA_DATE, STOCK_CODE, FACTOR_ID, FACTOR_VALUE) ' \
          'values(:1,:2,:3,:4)'
    flag=dataInsert(sql, value)
    return flag

def info_insert(value):
    sql = 'insert into ' + gl.TABLE_FACTOR_INFO + '(FACTOR_ID, FACTOR_NAME, STYLE_ID, STYLE_NAME) ' \
          'values(:1,:2,:3,:4)'
    flag=dataInsert(sql, value)
    return flag

def status_insert(value):
    sql = 'insert into ' + gl.TABLE_FACTOR_STATUS + '(WORK_ID, STATUS, ENTRY_TIME) ' \
                                                'values(:1,:2,:3)'
    flag = dataInsert(sql, value)
    return flag

# select s_info_windcode, trade_dt, s_dq_preclose from mktm.Aindexeodprices where s_info_windcode='000300.SH'
def get_stock_300(date_list):
    sql = 'select s_info_windcode, trade_dt, s_dq_preclose from ' + gl.TABLE_INDEXEOD_PRICES +  \
          ' where s_info_windcode=\'000300.SH\' intersect ' \
          'select s_info_windcode, trade_dt, s_dq_preclose from ' + gl.TABLE_INDEXEOD_PRICES +  \
          ' where trade_dt between %s and %s order by trade_dt desc' % (date_list[249], date_list[0])
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        stock_300 = pd.DataFrame()
    else:
        stock_300 = pd.DataFrame(result, index=list(index.iloc[:, 1]))
    return stock_300


# 获取当天的的个股收盘价(后复权)
def get_close_bak(cur_date):
    sql = 'select s_info_windcode,trade_dt,S_DQ_ADJCLOSE from ' + gl.TABLE_EOD_PRICES + \
          ' where trade_dt = %s' %(cur_date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        close_bak =pd.DataFrame()
    else:
        close_bak = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return close_bak



# 获取前12个月的个股收盘价(未复权)
def get_preclose(date_list):
    sql = 'select s_info_windcode,trade_dt,s_dq_preclose from ' + gl.TABLE_EOD_PRICES + \
          ' where trade_dt between %s and %s order by trade_dt desc' %(date_list[249], date_list[0])
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        preclose =pd.DataFrame()
    else:
        preclose = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return preclose


# 获取EP(TTM)，净利润/总市值(带index)，（若净利润<=0，则返回空）
def get_index_ep(date):
    sql = 'select s_info_windcode,trade_dt,s_val_pe_ttm from ' + gl.TABLE_EOD_DERIVATIVE  + '  \
                 where trade_dt=%s' % (date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        index_ep =pd.DataFrame()
    else:
        index_ep = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return index_ep


# 扣非后净利润，带index，-1列为rank=1列，-2列为扣非后净利润
def get_s_fa_deductedprofit(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_deductedprofit," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        s_fa_deductedprofit =pd.DataFrame()
    else:
        s_fa_deductedprofit = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return s_fa_deductedprofit


# 获取当日总市值(带index)
def get_index_mkvalue(date):
    sql = 'select s_info_windcode, trade_dt,s_val_mv from ' + gl.TABLE_EOD_DERIVATIVE + '  \
                 where trade_dt=%s' % (date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        index_mkvalue =pd.DataFrame()
    else:
        index_mkvalue = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return index_mkvalue


# 获取ps,市销率(PS,TTM)
def get_ps(date):
    sql = 'select s_info_windcode, trade_dt, s_val_ps_ttm from ' + gl.TABLE_EOD_DERIVATIVE + '  \
                 where trade_dt=%s' % (date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        ps =pd.DataFrame()
    else:
        ps = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return ps


# 获取净资产/总市值(带index)
def get_index_val_pb_new(date):
    sql = 'select s_info_windcode,trade_dt,s_val_pb_new from ' + gl.TABLE_EOD_DERIVATIVE + '  \
                 where trade_dt=%s' % (date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        val_pb_new =pd.DataFrame()
    else:
        val_pb_new = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return val_pb_new


# 获取营业收入(带index)
def get_index_oper_rev_ttm(date):
    sql = 'select s_info_windcode,trade_dt,oper_rev_ttm from ' + gl.TABLE_EOD_DERIVATIVE + '  \
                 where trade_dt=%s' % (date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        oper_rev_ttm = pd.DataFrame()
    else:
        oper_rev_ttm = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return oper_rev_ttm

# 获取净现金流(带index)
def get_index_val_pcf_ncfttm(date):
    sql = 'select s_info_windcode,trade_dt,s_val_pcf_ncfttm from ' + gl.TABLE_EOD_DERIVATIVE + '  \
                 where trade_dt=%s' % (date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        val_pcf_ncfttm = pd.DataFrame()
    else:
        val_pcf_ncfttm = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return val_pcf_ncfttm


# 获取经营性现金流(ttm)(带index)
def get_index_val_pcf_ocfttm(date):
    sql = 'select s_info_windcode,trade_dt,s_val_pcf_ocfttm from ' + gl.TABLE_EOD_DERIVATIVE + '  \
                 where trade_dt=%s' % (date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        val_pcf_ocfttm = pd.DataFrame()
    else:
        val_pcf_ocfttm = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return val_pcf_ocfttm


# 获取经营性现金流(lyr)(带index)
def get_index_val_pcf_ocf(date):
    sql = 'select s_info_windcode,trade_dt,s_val_pcf_ocf from ' + gl.TABLE_EOD_DERIVATIVE + '  \
                 where trade_dt=%s' % (date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        val_pcf_ocfttm = pd.DataFrame()
    else:
        val_pcf_ocfttm = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return val_pcf_ocfttm


# 单季度.净利润同比增长率(%)，-1列为rank,-2列为所需要的
def get_qfa_yoyprofit(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_qfa_yoyprofit," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        qfa_yoyprofit =pd.DataFrame()
    else:
        qfa_yoyprofit = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return qfa_yoyprofit


# 同比增长率-归属母公司股东的净利润-扣除非经常损益(%)，-1列为rank,-2列为所需要的
def get_profit_deducted(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_yoynetprofit_deducted," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        profit_deducted =pd.DataFrame()
    else:
        profit_deducted = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return profit_deducted


# 同比增长率-经营活动产生的现金流量净额(%)，-1列为rank,-2列为所需要的
def get_fa_yoyocf(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_yoyocf," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_yoyocf =pd.DataFrame()
    else:
        fa_yoyocf = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_yoyocf


# 扣除非经常损益后的净利润/净利润，-1列为rank,-2列为所需要的
def get_deductedprofittoprofit(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_deductedprofittoprofit," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        deductedprofittoprofit =pd.DataFrame()
    else:
        deductedprofittoprofit = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return deductedprofittoprofit


# 权益乘数，-1列为rank,-2列为所需要的
def get_fa_assetstoequity(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_assetstoequity," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_assetstoequity =pd.DataFrame()
    else:
        fa_assetstoequity = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_assetstoequity



# 单季度 营业收入环比增长率，-1列为rank,-2列为所需要的
def get_qfa_cgrsales(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_qfa_cgrsales," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        qfa_cgrsales =pd.DataFrame()
    else:
        qfa_cgrsales = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return qfa_cgrsales


# 营业收入同比增长率，-1列为rank,-2列为所需要的
def get_fa_yoy_or(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_yoy_or," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_yoy_or =pd.DataFrame()
    else:
        fa_yoy_or = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_yoy_or





# 单季度.净资产收益率，-1列为rank,-2列为所需要的
def get_qfa_roe(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_qfa_roe," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        qfa_roe =pd.DataFrame()
    else:
        qfa_roe = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return qfa_roe

# 净资产收益率(ttm)，-1列为rank,-2列为所需要的
def get_fa_roe(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_roe," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_roe =pd.DataFrame()
    else:
        fa_roe = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_roe


# 单季度.总资产净利润，-1列为rank,-2列为所需要的
def get_qfa_roa(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_qfa_roa," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        qfa_roa =pd.DataFrame()
    else:
        qfa_roa = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return qfa_roa


# 总资产净利润(ttm)，-1列为rank,-2列为所需要的
def get_fa_roa(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_roa," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_roa =pd.DataFrame()
    else:
        fa_roa = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_roa


# 单季度.销售毛利率，-1列为rank,-2列为所需要的
def get_qfa_grossprofitmargin(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_qfa_grossprofitmargin," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        qfa_grossprofitmargin =pd.DataFrame()
    else:
        qfa_grossprofitmargin = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return qfa_grossprofitmargin


# 销售毛利率(ttm)，-1列为rank,-2列为所需要的
def get_fa_grossprofitmargin(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_grossprofitmargin," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_grossprofitmargin =pd.DataFrame()
    else:
        fa_grossprofitmargin = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_grossprofitmargin

# 总资产周转率(ttm)，-1列为rank,-2列为所需要的
def get_fa_assetsturn(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_assetsturn," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_assetsturn =pd.DataFrame()
    else:
        fa_assetsturn = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_assetsturn


# 企业自由现金流，-1列为rank,-2列为所需要的
def get_fa_fcff(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_fcff," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_fcff =pd.DataFrame()
    else:
        fa_fcff = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_fcff


# 现金比率，-1列为rank,-2列为所需要的
def get_fa_cashtoliqdebt(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_cashtoliqdebt," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_cashtoliqdebt =pd.DataFrame()
    else:
        fa_cashtoliqdebt = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_cashtoliqdebt


# 流动比率，-1列为rank,-2列为所需要的
def get_fa_current(date):
    sql = "with a as " \
          "(select * from " \
          "(select distinct s_info_windcode, ann_dt,s_fa_current," \
          "rank() over (partition by s_info_windcode order by ann_dt desc,report_period desc ) rank from " \
          "" + gl.TABLE_FINANCIAL + " " \
          "where ann_dt<={}) aa  where aa.rank=1)," \
          "c as (select s_info_windcode from a group by s_info_windcode, ann_dt having count(*) >=1) " \
          "select a.* from a where exists(select s_info_windcode from c  where a.s_info_windcode = c.s_info_windcode)".format(date)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        fa_current =pd.DataFrame()
    else:
        fa_current = pd.DataFrame(result, index=list(index.iloc[:, 0]))
    return fa_current


# select s_dq_adjopen s_dq_adjclose from ' + TABLE_EOD_PRICES + '
# 获取N天交易日中复权开盘、收盘价，不带index，为了格式化df成为index为日期，columns为股票代码
def get_ndays_adjclose(date, trade_days):
    # date_list = get_n_trade_day(date, n)
    sql = 'select S_INFO_WINDCODE,TRADE_DT, s_dq_adjclose from ' + gl.TABLE_EOD_PRICES + ' ' \
          'where TRADE_DT <= %s and TRADE_DT >= %s order by TRADE_DT desc' % (trade_days[0], trade_days[-1])
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        ndays_adjclose = pd.DataFrame()
    else:
        ndays_adjclose = pd.DataFrame(result)
    return ndays_adjclose


# 获取N天交易日中每天的最高价，不带index，为了格式化df成为index为日期，columns为股票代码
def get_ndays_dq_high(date, trade_days):
    # date_list = get_n_trade_day(date, n)
    sql = 'select s_info_windcode,trade_dt, s_dq_high from ' + gl.TABLE_EOD_PRICES + ' ' \
          'where TRADE_DT <= %s and TRADE_DT >= %s order by TRADE_DT desc' % (trade_days[0], trade_days[-1])
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        dq_high = pd.DataFrame()
    else:
        dq_high = pd.DataFrame(result)
    return dq_high

# 获取N天交易日中每天的最低价，不带index，为了格式化df成为index为日期，columns为股票代码
def get_ndays_dq_low(date, trade_days):
    # date_list = get_n_trade_day(date, n)
    sql = 'select s_info_windcode,trade_dt, s_dq_low from ' + gl.TABLE_EOD_PRICES + ' ' \
          'where TRADE_DT <= %s and TRADE_DT >= %s order by TRADE_DT desc' % (trade_days[0], trade_days[-1])
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        dq_low = pd.DataFrame()
    else:
        dq_low = pd.DataFrame(result)
    return dq_low


# 获取换手率，不带index，为了格式化df成为index为日期，columns为股票代码
def get_dq_turn(date, trade_days):
    # date_list = get_n_trade_day(date, n)
    sql = 'select s_info_windcode,trade_dt, s_dq_turn from ' + gl.TABLE_EOD_DERIVATIVE + ' ' \
          'where TRADE_DT <= %s and TRADE_DT >= %s order by TRADE_DT desc' % (trade_days[0], trade_days[-1])
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        dq_turn = pd.DataFrame()
    else:
        dq_turn = pd.DataFrame(result)
    return dq_turn

# 获取成交量，不带index，为了格式化df成为index为日期，columns为股票代码
def get_dq_volume(date, trade_days):
    sql = 'select s_info_windcode,trade_dt, s_dq_volume from ' + gl.TABLE_EOD_PRICES + ' ' \
          'where TRADE_DT <= %s and TRADE_DT >= %s order by TRADE_DT desc' % (trade_days[0], trade_days[-1])
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        dq_volume = pd.DataFrame()
    else:
        dq_volume = pd.DataFrame(result)
    return dq_volume


# 获取当日流通股本，不带index，为了格式化df成为index为日期，columns为股票代码
def get_shr_today(date, trade_days):
    trade_days = tuple(trade_days)
    sql = 'select s_info_windcode,trade_dt, float_a_shr_today from ' + gl.TABLE_EOD_DERIVATIVE + ' ' \
          'where TRADE_DT <= %s and TRADE_DT >= %s order by TRADE_DT desc' % (trade_days[0], trade_days[-1])
    # sql = 'select s_info_windcode,trade_dt, float_a_shr_today from ' + gl.TABLE_EOD_DERIVATIVE + ' ' \
    #       'where TRADE_DT in {} order by TRADE_DT desc' .format(trade_days)
    result = dataSelect(sql)
    index = pd.DataFrame(result)
    if index.empty:
        shr_today = pd.DataFrame()
    else:
        shr_today = pd.DataFrame(result)
    return shr_today


def get_dps(start_date,end_date):
    sql='select t.s_info_windcode,a.dvd_sum,t.S_VAL_MV , a.dvd_sum/t.S_VAL_MV from ' + gl.TABLE_EOD_DERIVATIVE + '\
         t left join (select s_info_windcode,sum(d.cash_dvd_per_sh_pre_tax*d.s_div_baseshare) as \
         dvd_sum  from '+ gl.TABLE_DIVIDEND +' d where s_div_progress=3 and (ex_dt<={} and \
         ex_dt>={}) group by d.s_info_windcode) a on a.s_info_windcode=t.s_info_windcode \
         where  t.trade_dt={} and t.s_info_windcode is not null'.format(end_date,start_date,end_date)
    dp=pd.DataFrame(dataSelect(sql))
    return dp

#获取当天的所有股票列表
def get_all_stock(cur_date):
    sql='select distinct s_info_windcode from {} t where (s_info_delistdate is null or \
         s_info_delistdate >{}) and t.s_info_listdate<={}'.format(gl.TABLE_DESCRIPTION, cur_date, cur_date)
    stock_list=pd.DataFrame(dataSelect(sql))
    stock_list = stock_list[0].tolist()
    return stock_list














    


    
        
    