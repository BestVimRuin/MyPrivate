# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:57:20 2018

@author: 006702
"""
#风格因子ID
import logging
LOG_PATH = 'D:\\factor_matic_server.log'
logging.basicConfig(filename='{}'.format(LOG_PATH), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 用于指定stderr信息地址
workpath = 'D:\\gitee\\htsc_factor\\std\\'

SIZE_ID = 'S000001'
QUALITY_ID ='S000002'
HEAT_ID ='S000003'
VOLATILITY_ID = 'S000004'
MOMENTUM_ID = 'S000005'
GROWTH_ID = 'S000006'
NON_STYLE_ID = 'S000000'


SELF_SELECTED_STOCK_FLAG=0
MODEL_STOCK_FLAG=1

SELF_SELECTED_STOCK_ID='ZXGRroup'
SELF_SELECTED_STOCK_NAME='自选股'
MODEL_GROUP_ID='ModelGroup001'
MODEL_GROUP_NAME='模型组合'
BENCHMARK_HS300='HS300'
ALL_MARKET_STOCK='ALLSTOCK'
NEUTRAL_GROUP_TYPE=''


FOF_NAME = 'fof_style'
FOF_PWD = 'Style_analysis'
FOF_PORT = '168.61.13.175:1521/ost'

TABLE_CALENDAR = 'mktm.asharecalendar'
TABLE_DIVIDEND = 'mktm.asharedividend'
TABLE_EOD_DERIVATIVE = 'mktm.ashareeodderivativeindicator'
TABLE_EOD_PRICES = 'mktm.ashareeodprices'
TABLE_FINANCIAL = 'mktm.asharefinancialindicator'
TABLE_INDEXEOD_PRICES = 'mktm.aindexeodprices'
TABLE_DESCRIPTION = 'mktm.asharedescription'

# fof用于插入的表
TABLE_FACTOR_STATUS = 'M_TABLE_STATUS_TEST'
TABLE_FACTOR_DATA = 'M_FACTOR_DATA_TEST'
TABLE_FACTOR_INFO = 'M_FACTOR_INFO_TEST'
TABLE_FACTOR_MODEL = 'M_FACTOR_MODEL_TEST'
