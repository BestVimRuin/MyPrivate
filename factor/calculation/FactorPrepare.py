# -*- coding: utf-8 -*-
import time
from datetime import date
from dateutil.parser import parse
import numpy as np
import pandas as pd
from sklearn import linear_model
from dao import OracleManager as om
from util import FormatHandler as fh
from util import GlobalList as gl


# 定义每组因子
def factor_value(cur_date):
    # 获取EP(TTM)，净利润/总市值(带index)，（若净利润<=0，则返回空）
    inverse_ep = om.get_index_ep(cur_date)
    # 获取ps,市销率(PS,TTM)
    ps = om.get_ps(cur_date)
    # 获取pb值  也就是需要的bp，的倒数
    inverse_bp = om.get_index_val_pb_new(cur_date)
    # 获取净现金流
    val_pcf_ncfttm = om.get_index_val_pcf_ncfttm(cur_date)
    # 获取经营性现金流(ttm)
    val_pcf_ocfttm = om.get_index_val_pcf_ocfttm(cur_date)
    # 获取个股市值(带index)
    index_mkvalue = om.get_index_mkvalue(cur_date)

    ep = 1 / inverse_ep.iloc[:, -1]
    bp = 1 / inverse_bp.iloc[:, -1]
    sp = 1 / ps.iloc[:, -1]
    ncfp = val_pcf_ncfttm.iloc[:, -1] / index_mkvalue.iloc[:, -1]
    ocfp = val_pcf_ocfttm.iloc[:, -1] / index_mkvalue.iloc[:, -1]
    # 分红放在最后面了,主要是不是我写的
    # 红利(近12个月股息率),首先拼接一年前的日期
    struct_time = time.strptime(str(cur_date), '%Y%m%d')
    y = struct_time.tm_year - 1
    m = struct_time.tm_mon
    d = struct_time.tm_mday
    year_before = str(date(y, m, d).replace(year=y))
    year_before = year_before.replace('-', '')
    divdend = om.get_dps(year_before, str(cur_date))
    divdend.index = divdend.iloc[:, 0]
    dp = divdend.iloc[:, -1]
    # 入库
    stock_list = om.get_all_stock(cur_date)
    ep_flag = fh.df_to_list(cur_date, ep, 'ep', stock_list)
    bp_flag = fh.df_to_list(cur_date, bp, 'bp', stock_list)
    sp_flag = fh.df_to_list(cur_date, sp, 'sp', stock_list)
    ncfp_flag = fh.df_to_list(cur_date, ncfp, 'ncfp', stock_list)
    ocfp_flag = fh.df_to_list(cur_date, ocfp, 'ocfp', stock_list)
    dp_flag = fh.df_to_list(cur_date, dp, 'dp', stock_list)

    if ep_flag & bp_flag & sp_flag & ncfp_flag & ocfp_flag & dp_flag != 1:
        gl.logger.error('factor_value insert fail')
        return 0
    else:
        gl.logger.info('factor_value insert successful')
        return 1


def factor_growth(cur_date):
    # 单季度 营业收入环比增长率，-1列为rank,-2列为所需要的
    qfa_cgrsales = om.get_qfa_cgrsales(cur_date)
    # 营业收入同比增长率，-1列为rank,-2列为所需要的
    fa_yoy_or = om.get_fa_yoy_or(cur_date)
    # 单季度.净利润同比增长率(%)，-1列为rank,-2列为所需要的
    qfa_yoyprofit = om.get_qfa_yoyprofit(cur_date)
    # 同比增长率-归属母公司股东的净利润-扣除非经常损益(%)，-1列为rank,-2列为所需要的
    profit_deducted = om.get_profit_deducted(cur_date)
    # 同比增长率-经营活动产生的现金流量净额(%)，-1列为rank,-2列为所需要的
    fa_yoyocf = om.get_fa_yoyocf(cur_date)
    # 同比增长率-归属母公司股东的净利润-扣除非经常损益(%)，-1列为rank,-2列为所需要的
    profit_deducted_second = om.get_profit_deducted(cur_date - 10000)
    # 同比增长率-归属母公司股东的净利润-扣除非经常损益(%)，-1列为rank,-2列为所需要的
    profit_deducted_third = om.get_profit_deducted(cur_date - 20000)
    # 同比增长率-经营活动产生的现金流量净额(%)，-1列为rank,-2列为所需要的
    fa_yoyocf_second = om.get_fa_yoyocf(cur_date - 10000)
    # 同比增长率-经营活动产生的现金流量净额(%)，-1列为rank,-2列为所需要的
    fa_yoyocf_third = om.get_fa_yoyocf(cur_date - 20000)
    # 营业收入同比增长率，-1列为rank,-2列为所需要的(去年的)
    fa_yoy_or_second = om.get_fa_yoy_or(cur_date - 10000)
    # 营业收入同比增长率，-1列为rank,-2列为所需要的(前年的)
    fa_yoy_or_third = om.get_fa_yoy_or(cur_date - 20000)

    sales_growth_q = qfa_cgrsales.iloc[:, -2]
    sales_growth_ttm = fa_yoy_or.iloc[:, -2]
    sales_growth_3y = (((fa_yoy_or.iloc[:, -2] / 100 + 1) * (fa_yoy_or_second.iloc[:, -2] / 100 + 1) *
                       (fa_yoy_or_third.iloc[:, -2] / 100 + 1)) ** (1 / 3) - 1) * 100
    profit_growth_q = qfa_yoyprofit.iloc[:, -2]
    profit_growth_ttm = profit_deducted.iloc[:, -2]
    profit_growth_3y = (((profit_deducted.iloc[:, -2] / 100 + 1) * (profit_deducted_second.iloc[:, -2] / 100 + 1) *
                        (profit_deducted_third.iloc[:, -2] / 100 + 1)) ** (1 / 3) - 1) * 100
    operationcashflow_growth_ttm = fa_yoyocf.iloc[:, -2]
    operationcashflow_growth_3y = (((fa_yoyocf.iloc[:, -2] / 100 + 1) * (fa_yoyocf_second.iloc[:, -2] / 100 + 1) *
                                   (fa_yoyocf_third.iloc[:, -2] / 100 + 1)) ** (1 / 3) - 1) * 100
    # 入库
    stock_list = om.get_all_stock(cur_date)
    sales_growth_q_flag = fh.df_to_list(cur_date, sales_growth_q, 'sales_growth_q', stock_list)
    sales_growth_ttm_flag = fh.df_to_list(cur_date, sales_growth_ttm, 'sales_growth_ttm', stock_list)
    sales_growth_3y_flag = fh.df_to_list(cur_date, sales_growth_3y, 'sales_growth_3y', stock_list)
    profit_growth_q_flag = fh.df_to_list(cur_date, profit_growth_q, 'profit_growth_q', stock_list)
    profit_growth_ttm_flag = fh.df_to_list(cur_date, profit_growth_ttm, 'profit_growth_ttm', stock_list)
    profit_growth_3y_flag = fh.df_to_list(cur_date, profit_growth_3y, 'profit_growth_3y', stock_list)
    operationcashflow_growth_ttm_flag = fh.df_to_list(cur_date, operationcashflow_growth_ttm,
                                                      'operationcashflow_growth_ttm', stock_list)
    operationcashflow_growth_3y_flag = fh.df_to_list(cur_date, operationcashflow_growth_3y,
                                                     'operationcashflow_growth_3y', stock_list)

    if sales_growth_q_flag & sales_growth_ttm_flag & sales_growth_3y_flag & profit_growth_q_flag & profit_growth_ttm_flag & \
            profit_growth_3y_flag & operationcashflow_growth_ttm_flag & operationcashflow_growth_3y_flag != 1:
        gl.logger.error('factor_growth insert fail')
        return 0
    else:
        gl.logger.info('factor_growth insert successful')
        return 1


def factor_financial_quality(cur_date):
    # 获取经营性现金流(ttm)
    val_pcf_ocfttm = om.get_index_val_pcf_ocfttm(cur_date)
    # 单季度.净资产收益率，-1列为rank,-2列为所需要的
    qfa_roe = om.get_qfa_roe(cur_date)
    # 净资产收益率(ttm)，-1列为rank,-2列为所需要的
    fa_roe = om.get_fa_roe(cur_date)
    # 单季度.总资产净利润，-1列为rank,-2列为所需要的
    qfa_roa = om.get_qfa_roa(cur_date)
    # 总资产净利润(ttm)，-1列为rank,-2列为所需要的
    fa_roa = om.get_fa_roa(cur_date)
    # 单季度.销售毛利率，-1列为rank,-2列为所需要的
    qfa_grossprofitmargin = om.get_qfa_grossprofitmargin(cur_date)
    # 销售毛利率(ttm)，-1列为rank,-2列为所需要的
    fa_grossprofitmargin = om.get_fa_grossprofitmargin(cur_date)
    # 扣除非经常损益后的净利润/净利润，-1列为rank,-2列为所需要的
    deductedprofittoprofit = om.get_deductedprofittoprofit(cur_date)
    # 总资产周转率(ttm)，-1列为rank,-2列为所需要的
    fa_assetsturn = om.get_fa_assetsturn(cur_date)
    # 获取经营性现金流(lyr)
    val_pcf_ocf = om.get_index_val_pcf_ocf(cur_date)

    roe_q = qfa_roe.iloc[:, -2]
    roe_ttm = fa_roe.iloc[:, -2]
    roa_q = qfa_roa.iloc[:, -2]
    roa_ttm = fa_roa.iloc[:, -2]
    grossprofitmargin_q = qfa_grossprofitmargin.iloc[:, -2]
    grossprofitmargin_ttm = fa_grossprofitmargin.iloc[:, -2]
    profitmargin_ttm = deductedprofittoprofit.iloc[:, -2]
    assetturnover_ttm = fa_assetsturn.iloc[:, -2]
    operationcashflowratio_q = val_pcf_ocf.iloc[:, -1] / qfa_roa.iloc[:, -2]
    operationcashflowratio_ttm = val_pcf_ocfttm.iloc[:, -1] / fa_roa.iloc[:, -2]
    # 入库
    stock_list = om.get_all_stock(cur_date)
    roe_q_flag = fh.df_to_list(cur_date, roe_q, 'roe_q', stock_list)
    roe_ttm_flag = fh.df_to_list(cur_date, roe_ttm, 'roe_ttm', stock_list)
    roa_q_flag = fh.df_to_list(cur_date, roa_q, 'roa_q', stock_list)
    roa_ttm_flag = fh.df_to_list(cur_date, roa_ttm, 'roa_ttm', stock_list)
    grossprofitmargin_q_flag = fh.df_to_list(cur_date, grossprofitmargin_q, 'grossprofitmargin_q', stock_list)
    grossprofitmargin_ttm_flag = fh.df_to_list(cur_date, grossprofitmargin_ttm, 'grossprofitmargin_ttm', stock_list)
    profitmargin_ttm_flag = fh.df_to_list(cur_date, profitmargin_ttm, 'profitmargin_ttm', stock_list)
    assetturnover_ttm_flag = fh.df_to_list(cur_date, assetturnover_ttm, 'assetturnover_ttm', stock_list)
    operationcashflowratio_q_flag = fh.df_to_list(cur_date, operationcashflowratio_q, 'operationcashflowratio_q', stock_list)
    operationcashflowratio_ttm_flag = fh.df_to_list(cur_date, operationcashflowratio_ttm, 'operationcashflowratio_ttm', stock_list)

    if roe_q_flag & roe_ttm_flag & roa_q_flag & roa_ttm_flag & grossprofitmargin_q_flag & grossprofitmargin_ttm_flag \
            & profitmargin_ttm_flag & assetturnover_ttm_flag & operationcashflowratio_q_flag & operationcashflowratio_ttm_flag != 1:
        gl.logger.error('factor_financial_quality insert fail')
        return 0
    else:
        gl.logger.info('factor_financial_quality insert successful')
        return 1


def factor_leverage(cur_date):
    # 现金比率，-1列为rank,-2列为所需要的
    fa_cashtoliqdebt = om.get_fa_cashtoliqdebt(cur_date)
    # 流动比率，-1列为rank,-2列为所需要的
    fa_current = om.get_fa_current(cur_date)
    # 权益乘数，-1列为rank,-2列为所需要的
    fa_assetstoequity = om.get_fa_assetstoequity(cur_date)
    # 获取个股市值(带index)
    index_mkvalue = om.get_index_mkvalue(cur_date)

    cashration = fa_cashtoliqdebt.iloc[:, -2]  # 现金比率
    currentratio = fa_current.iloc[:, -2]  # 流动比率
    equitymultiplier = fa_assetstoequity.iloc[:, -2]
    ln_capital = np.log(index_mkvalue.iloc[:, -1])  # 市值对数
    # 入库
    stock_list = om.get_all_stock(cur_date)
    cashration_flag = fh.df_to_list(cur_date, cashration, 'cashration', stock_list)
    currentratio_flag = fh.df_to_list(cur_date, currentratio, 'currentratio', stock_list)
    equitymultiplier_flag = fh.df_to_list(cur_date, equitymultiplier, 'equitymultiplier', stock_list)
    ln_capital_flag = fh.df_to_list(cur_date, ln_capital, 'ln_capital', stock_list)

    if cashration_flag & currentratio_flag & equitymultiplier_flag & ln_capital_flag != 1:
        gl.logger.error('factor_leverage insert fail')
        return 0
    else:
        gl.logger.info('factor_leverage insert successful')
        return 1


def factor_momentum(cur_date, regress_result, date_adjclose):
    HAlpha = regress_result.loc['const']
    relative_strength_1m = date_adjclose.iloc[0] / date_adjclose.shift(-20).iloc[0] - 1
    relative_strength_2m = date_adjclose.iloc[0] / date_adjclose.shift(-20 * 2).iloc[0] - 1
    relative_strength_3m = date_adjclose.iloc[0] / date_adjclose.shift(-20 * 3).iloc[0] - 1
    relative_strength_6m = date_adjclose.iloc[0] / date_adjclose.shift(-20 * 6).iloc[0] - 1
    relative_strength_12m = date_adjclose.iloc[0] / date_adjclose.shift(-20 * 12).iloc[0] - 1
    # 入库
    stock_list = om.get_all_stock(cur_date)
    HAlpha_flag = fh.df_to_list(cur_date, HAlpha, 'HAlpha', stock_list)
    relative_strength_1m_flag = fh.df_to_list(cur_date, relative_strength_1m, 'relative_strength_1m', stock_list)
    relative_strength_2m_flag = fh.df_to_list(cur_date, relative_strength_2m, 'relative_strength_2m', stock_list)
    relative_strength_3m_flag = fh.df_to_list(cur_date, relative_strength_3m, 'relative_strength_3m', stock_list)
    relative_strength_6m_flag = fh.df_to_list(cur_date, relative_strength_6m, 'relative_strength_6m', stock_list)
    relative_strength_12m_flag = fh.df_to_list(cur_date, relative_strength_12m, 'relative_strength_12m', stock_list)

    if HAlpha_flag & relative_strength_1m_flag & relative_strength_2m_flag & relative_strength_3m_flag & \
            relative_strength_6m_flag & relative_strength_12m_flag != 1:
        gl.logger.error('factor_momentum insert fail')
        return 0
    else:
        gl.logger.info('factor_momentum insert successful')
        return 1


def factor_volatility(cur_date, date_list, regress_result, date_adjclose):
    m1, m2, m3, m6, m12 = 20, 40, 60, 120, 240
    # 获取N天交易日中每天的最高价和最低价，不带index，为了格式化df成为index为日期，columns为股票代码
    price_high = om.get_ndays_dq_high(cur_date, date_list)
    price_low = om.get_ndays_dq_low(cur_date, date_list)
    # 得到近一年的最高价,最低价,并转换为index为日期，columns为股票代码
    date_price_high = fh.transform_large_df(price_high)
    date_price_low = fh.transform_large_df(price_low)
    # 降序排列index（最上面的是最新的日期）, 结果为20天内最高价和最低价
    # 获取N天交易日中每天的最高价，不带index，为了格式化df成为index为日期，columns为股票代码
    price_max_1m, price_min_1m = price(date_price_high, date_price_low, m1)
    price_max_2m, price_min_2m = price(date_price_high, date_price_low, m2)
    price_max_3m, price_min_3m = price(date_price_high, date_price_low, m3)
    price_max_6m, price_min_6m = price(date_price_high, date_price_low, m6)
    price_max_12m, price_min_12m = price(date_price_high, date_price_low, m12)

    high_low_1m = price_max_1m / price_min_1m
    high_low_2m = price_max_2m / price_min_2m
    high_low_3m = price_max_3m / price_min_3m
    high_low_6m = price_max_6m / price_min_6m
    high_low_12m = price_max_12m / price_min_12m
    std_1m = std_adjclose_rate(date_adjclose, m1)
    std_2m = std_adjclose_rate(date_adjclose, m2)
    std_3m = std_adjclose_rate(date_adjclose, m3)
    std_6m = std_adjclose_rate(date_adjclose, m6)
    std_12m = std_adjclose_rate(date_adjclose, m12)
    ln_price = np.log(date_adjclose.iloc[0])  # 用的每日收盘价
    beta_consistence = regress_result.loc['beta_consis']
    # 入库
    stock_list = om.get_all_stock(cur_date)
    high_low_1m_flag = fh.df_to_list(cur_date, high_low_1m, 'high_low_1m', stock_list)
    high_low_2m_flag = fh.df_to_list(cur_date, high_low_2m, 'high_low_2m', stock_list)
    high_low_3m_flag = fh.df_to_list(cur_date, high_low_3m, 'high_low_3m', stock_list)
    high_low_6m_flag = fh.df_to_list(cur_date, high_low_6m, 'high_low_6m', stock_list)
    high_low_12m_flag = fh.df_to_list(cur_date, high_low_12m, 'high_low_12m', stock_list)
    std_1m_flag = fh.df_to_list(cur_date, std_1m, 'std_1m', stock_list)
    std_2m_flag = fh.df_to_list(cur_date, std_2m, 'std_2m', stock_list)
    std_3m_flag = fh.df_to_list(cur_date, std_3m, 'std_3m', stock_list)
    std_6m_flag = fh.df_to_list(cur_date, std_6m, 'std_6m', stock_list)
    std_12m_flag = fh.df_to_list(cur_date, std_12m, 'std_12m', stock_list)
    ln_price_flag = fh.df_to_list(cur_date, ln_price, 'ln_price', stock_list)
    beta_consistence_flag = fh.df_to_list(cur_date, beta_consistence, 'beta_consistence', stock_list)

    if high_low_1m_flag & high_low_2m_flag & high_low_3m_flag & high_low_6m_flag & high_low_12m_flag & \
            std_1m_flag & std_2m_flag & std_3m_flag & std_6m_flag & std_12m_flag & ln_price_flag & beta_consistence_flag != 1:
        gl.logger.error('factor_volatility insert fail')
        return 0
    else:
        gl.logger.info('factor_volatility insert successful')
        return 1


def factor_turnover(cur_date, date_list):
    m1, m2, m3, m6, m12 = 20, 40, 60, 120, 240
    # 此处为了获得换手率,取得成交量和流通股本
    # 获取成交量，和当日流通股本,不带index，为了格式化df成为index为日期，columns为股票代码
    dq_volume = om.get_dq_volume(cur_date, date_list)
    shr_today = om.get_shr_today(cur_date, date_list)
    # 得到近一年的最高价,最低价,并转换为index为日期，columns为股票代码
    date_dq_volume = fh.transform_large_df(dq_volume)
    date_shr_today = fh.transform_large_df(shr_today)

    turnover = date_dq_volume / date_shr_today
    turnover = turnover.loc[list(date_dq_volume.index)]
    # 在中国：换手率= 成交量/流通股本×100%,炒股软件的成交量单位是万,流通股本单位是亿
    # 流通股本基本不变(大概1年变一次),我都取得各自的总量,然后相除
    format = lambda x: '%.4f' % x
    turnover_1m = turnover[:m1].sum()
    turnover_2m = turnover[:m2].sum()
    turnover_3m = turnover[:m3].sum()
    turnover_6m = turnover[:m6].sum()
    turnover_12m = turnover[:m12].sum()

    turnover_1m = turnover_1m[~turnover_1m.isin([np.nan, np.inf, -np.inf])]
    turnover_2m = turnover_2m[~turnover_2m.isin([np.nan, np.inf, -np.inf])]
    turnover_3m = turnover_3m[~turnover_3m.isin([np.nan, np.inf, -np.inf])]
    turnover_6m = turnover_6m[~turnover_6m.isin([np.nan, np.inf, -np.inf])]
    turnover_12m = turnover_12m[~turnover_12m.isin([np.nan, np.inf, -np.inf])]

    turnover_1m = turnover_1m.dropna()
    turnover_2m = turnover_2m.dropna()
    turnover_3m = turnover_3m.dropna()
    turnover_6m = turnover_6m.dropna()
    turnover_12m = turnover_12m.dropna()

    turnover_1m = turnover_1m.map(format)
    turnover_2m = turnover_2m.map(format)
    turnover_3m = turnover_3m.map(format)
    turnover_6m = turnover_6m.map(format)
    turnover_12m = turnover_12m.map(format)

    # 入库
    stock_list = om.get_all_stock(cur_date)
    turnover_1m_flag = fh.df_to_list(cur_date, turnover_1m, 'turnover_1m', stock_list)
    turnover_2m_flag = fh.df_to_list(cur_date, turnover_2m, 'turnover_2m', stock_list)
    turnover_3m_flag = fh.df_to_list(cur_date, turnover_3m, 'turnover_3m', stock_list)
    turnover_6m_flag = fh.df_to_list(cur_date, turnover_6m, 'turnover_6m', stock_list)
    turnover_12m_flag = fh.df_to_list(cur_date, turnover_12m, 'turnover_12m', stock_list)

    if turnover_1m_flag & turnover_2m_flag & turnover_3m_flag & turnover_6m_flag & turnover_12m_flag != 1:
        gl.logger.error('factor_turnover insert fail')
        return 0
    else:
        gl.logger.info('factor_turnover insert successful')
        return 1


def factor_modified_momentum(cur_date, date_list, date_adjclose, date_dq_turn):
    m1, m2, m3, m6, m12 = 20, 40, 60, 120, 240
    weighted_strength_1m = calculate_rate(date_adjclose, date_dq_turn, m1)
    weighted_strength_2m = calculate_rate(date_adjclose, date_dq_turn, m2)
    weighted_strength_3m = calculate_rate(date_adjclose, date_dq_turn, m3)
    weighted_strength_6m = calculate_rate(date_adjclose, date_dq_turn, m6)
    weighted_strength_12m = calculate_rate(date_adjclose, date_dq_turn, m12)
    # 入库
    stock_list = om.get_all_stock(cur_date)
    weighted_strength_1m_flag = fh.df_to_list(cur_date, weighted_strength_1m, 'weighted_strength_1m', stock_list)
    weighted_strength_2m_flag = fh.df_to_list(cur_date, weighted_strength_2m, 'weighted_strength_2m', stock_list)
    weighted_strength_3m_flag = fh.df_to_list(cur_date, weighted_strength_3m, 'weighted_strength_3m', stock_list)
    weighted_strength_6m_flag = fh.df_to_list(cur_date, weighted_strength_6m, 'weighted_strength_6m', stock_list)
    weighted_strength_12m_flag = fh.df_to_list(cur_date, weighted_strength_12m, 'weighted_strength_12m', stock_list)

    if weighted_strength_1m_flag & weighted_strength_2m_flag & weighted_strength_3m_flag & weighted_strength_6m_flag & \
            weighted_strength_12m_flag != 1:
        gl.logger.error('factor_modified_momentum insert fail')
        return 0
    else:
        gl.logger.info('factor_modified_momentum insert successful')
        return 1


def factor_close_bak(cur_date):
    close_bak = om.get_close_bak(cur_date)
    close_bak = close_bak.iloc[:, -1]
    stock_list = om.get_all_stock(cur_date)
    close_bak_flag = fh.df_to_list(cur_date, close_bak, 'close_bak', stock_list)
    if close_bak_flag != 1:
        gl.logger.error('factor_close_bak insert fail')
        return 0
    else:
        gl.logger.info('factor_close_bak insert successful')
        return 1


# 返回的是这个日期的季度的第一天和三个月前的那个交易日()
def date_handle(date):
    format_date = parse(str(date))  # 格式化为2016-10-02 00:00:00
    year = format_date.year
    month = format_date.month
    day = format_date.day
    quarter_fm = month - (month - 1) % 3  # quarter_first_month该月份所属季度的第一个月
    if quarter_fm < 10:
        quarter_fm = '0' + str(quarter_fm)
    quarter_start = str(year) + str(quarter_fm) + '01'
    #  date_3month 3个月前的日期
    if month < 3:
        year -= 1
        month = month + 9  # 其实就是+12-3，借了一年（12个月）再减去3个月
    else:
        month -= 3
    if month < 10:
        month = '0' + str(month)
    if day < 10:
        day = '0' + str(day)
    date_3month = str(year) + str(month) + str(day)
    return quarter_start, date_3month  # 返回的是这个日期的季度的第一天和三个月前的那个交易日()


# 格式化最高价,最低价
def price(date_price_high, date_price_low, n):
    gl.logger.info('price func start')
    price_n_high = date_price_high[0:n]
    price_n_low = date_price_low[0:n]
    date_price_max = price_n_high.max()
    date_price_min = price_n_low.min()
    return date_price_max, date_price_min


# 日收益率标准差
def std_adjclose_rate(date_adjclose, n):
    gl.logger.info('std_adjclose_rate func start')
    n_days_adjclose = date_adjclose[0:n + 1]
    n_days_rate = n_days_adjclose / n_days_adjclose.shift(-1)
    std_rate = n_days_rate.std()
    return std_rate


# 换手率
def dq_turn_total(date_dq_volume, date_shr_today, n):
    gl.logger.info('dq_turn_total func start')
    dq_volume_n = date_dq_volume[0:n].sum()
    shr_today_n = date_shr_today[0:n].sum()
    return dq_volume_n, shr_today_n


def operate_dq_turn(date_dq_turn):
    gl.logger.info('operate_dq_turn func start')
    dq_turn_total = date_dq_turn.sum()
    weight_dq_turn = date_dq_turn / dq_turn_total
    return weight_dq_turn


# 和上面的一起的,算的是换手率加权日收益率
def calculate_rate(date_adjclose, date_dq_turn, n):
    gl.logger.info('calculate_rate func start')
    index_l = []
    adjclose_n = date_adjclose.iloc[0:n + 1]
    adjclose_rate = adjclose_n / adjclose_n.shift(-1) - 1
    adjclose_rate.drop(adjclose_rate.index[n], axis=0, inplace=True)
    date_dq_turn = date_dq_turn.iloc[0:n]
    weight_dq_turn = operate_dq_turn(date_dq_turn)
    for i in weight_dq_turn.index:
        index_l.append(str(i))
    weight_dq_turn.index = index_l
    rate_result = adjclose_rate * 100 * weight_dq_turn
    date_rate_result = rate_result.sum() / n
    return date_rate_result


# 12个月的个股收益对沪深300  线性回归
def stock_regress(cur_date, date_list):
    # 获取前12个月的个股收盘价(未复权)
    preclose = om.get_preclose(date_list)
    preclose = fh.transform_large_df(preclose)
    # 求收益率,并且去掉最后一行
    preclose = preclose / preclose.shift(-1) - 1
    # 此处为行是日期,列为股票代码
    preclose = preclose.iloc[:-1]

    # 获取沪深300指数(未复权),取出来是一个数
    stock_300 = om.get_stock_300(date_list)
    stock_300 = stock_300.iloc[:, -1]
    stock_300 = stock_300 / stock_300.shift(-1) - 1
    stock_300 = stock_300.iloc[:-1]

    train = pd.concat([stock_300, preclose], axis=1, join='inner')
    train.fillna(0, inplace=True)
    x_df = train.iloc[:, 0:1]
    y_df = train.iloc[:, 1:]
    model = linear_model.LinearRegression()
    regress_result = {}    
    for y_col_num in range(y_df.shape[1]):
        y_matrix = np.mat(y_df.iloc[:, y_col_num]).T
        x_matrix = np.mat(x_df)
        reg = model.fit(x_matrix, y_matrix)
        coef = reg.coef_[0][0]
        const = reg.intercept_[0]
        beta_consis = coef * (y_matrix - np.dot(coef, x_matrix) + const).std()
        regress_result[y_df.columns[y_col_num]] = [coef, const, beta_consis]
    return regress_result


def factor_prepare(cur_date):
    gl.logger.info("start to prepare factor data----------")
    n = -365  # 主要因为dq_turn会返回周六周日的日期,数据为na,但是占了行数
    # 取某个日期前(负，-7)or后(正，+7)N天的交易日日期,date_list[0]为最近的一天
    date_list = om.getNtradeDate(cur_date, n)
    
    # 12个月的个股收益对沪深300  线性回归
    gl.logger.info('start to prepare regress data')
    regress_result = stock_regress(cur_date, date_list)
    regress_result = pd.DataFrame.from_dict(regress_result, orient='columns')
    # index分别为权重,常数项,经过处理的残差
    regress_result.index = ['coef', 'const', 'beta_consis']

    # 获取500天的复权收益列表,为降序排列,第0个位最新的
    gl.logger.info('start to prepare adjclose data')
    adjclose = om.get_ndays_adjclose(cur_date, date_list)
    # 格式化index为日期,columns为股票代码,降序,第0个位最新的
    date_adjclose = fh.transform_large_df(adjclose)

    # 换手率加权*日均收益率
    # 获取换手率，不带index，为了格式化df成为index为日期，columns为股票代码
    gl.logger.info('start to prepare dq_turn data')
    dq_turn = om.get_dq_turn(cur_date, date_list)
    date_dq_turn = fh.transform_large_df(dq_turn)
    # 这个会返回周六周日,数据为na,所以导致n取365
    date_dq_turn.dropna(how='all', inplace=True)

    # -----------------------------------------------------------------------------
    #     估值因子,value factor
    gl.logger.info('start to run factor_value function')
    factor_value_flag = factor_value(cur_date)

    # -----------------------------------------------------------------------------
    #     成长因子,growth factor
    gl.logger.info('start to run factor_growth function')
    factor_growth_flag = factor_growth(cur_date)

    # -----------------------------------------------------------------------------
    #     财务质量因子,financial quality factor
    gl.logger.info('start to run factor_financial_quality function')
    factor_financial_flag = factor_financial_quality(cur_date)

    # -----------------------------------------------------------------------------
    #     杠杠因子,leverage factor
    #     规模因子,size factor
    gl.logger.info('start to run factor_leverage function')
    factor_leverage_flag = factor_leverage(cur_date)

    # -----------------------------------------------------------------------------
    #     动量因子,momentum factor
    gl.logger.info('start to run factor_momentum function')
    factor_momentum_flag = factor_momentum(cur_date, regress_result, date_adjclose)

    # -----------------------------------------------------------------------------
    #     波动率因子,volatility factor
    gl.logger.info('start to run factor_volatility function')
    factor_volatility_flag = factor_volatility(cur_date, date_list, regress_result, date_adjclose)

    # -----------------------------------------------------------------------------
    #     换手率因子,turnover factor
    gl.logger.info('start to run factor_turnover function')
    factor_turnover_flag = factor_turnover(cur_date, date_list)
    #     factor_turnover_flag = 1

    # -----------------------------------------------------------------------------
    #     改进的动量因子,modified momentum factor
    gl.logger.info('start to run factor_modified_momentum function')
    factor_modified_flag = factor_modified_momentum(cur_date, date_list, date_adjclose, date_dq_turn)

    # -----------------------------------------------------------------------------
    gl.logger.info('start to run factor_close_bak function')
    close_bak_flag = factor_close_bak(cur_date)

    flag = factor_value_flag & factor_growth_flag & factor_financial_flag & factor_leverage_flag & factor_momentum_flag \
            & factor_volatility_flag & factor_turnover_flag & factor_modified_flag & close_bak_flag

    # 所有函数的返回值是不是都成功了
    work_time  = time.strftime('%Y%m%d%H%M%S')
    work_id = 'FactorPrepare_{}'.format(work_time)
    entry_time = time.strftime('%Y-%m-%d %H:%M:%S')
    if flag != 1:
        status_flag = om.status_insert([[work_id, 0, entry_time]])
        gl.logger.error('some function are  failed----------')
        return 0
    else:
        status_flag = om.status_insert([[work_id, 1, entry_time]])
        gl.logger.info('all function are successful----------')
        return 1




if __name__ == '__main__':
    cur_date = 20190109
    st_time = time.time()
    factor_prepare_flag = factor_prepare(cur_date)
    print(time.time()  - st_time)




