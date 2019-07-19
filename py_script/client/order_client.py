# -*- coding: utf-8 -*-
import os
import zmq
import sys
import json
import numpy as np
import logging
import datetime

# from keras.models import load_model

dirs = 'logs'
if not os.path.exists(dirs):
    os.makedirs(dirs)

logging.basicConfig(filename='%s/info_cl.log' % dirs, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
_orders = {}


# class orderDict:
#
#    def __init__(slef,):
#        self.clOrderID 

class order_client:
    sum_order_qty = 0	# 命令总数量
    sum_cancel_qty = 0	# 取消的总数量

    def __init__(self, xquantid):
        # 主要用zmq去连接服务器，具体如何配置还需要研究
        self.xquantid = xquantid
        port = 20000 + self.xquantid % 10000
        context = zmq.Context()
        self.order_client = context.socket(zmq.REQ)         # 链接服务器的固定套路,zmq.REQ is clint and zmq.REP is sever
        self.order_client.connect("tcp://localhost:%s" % str(port))   #  why connect 3 counts ,order_client order_manager(2)

        port = 30000 + self.xquantid % 10000
        context = zmq.Context()
        self.order_manager = context.socket(zmq.SUB)
        self.order_manager.connect("tcp://localhost:%s" % str(port))  # 接order
        self.order_manager.connect("tcp://168.61.32.91:9555")  # 接行情
        self.order_manager.setsockopt_string(zmq.SUBSCRIBE, self.stock)     # what is this
        self.order_manager.setsockopt_string(zmq.SUBSCRIBE, 'Trade')
        self.order_manager.setsockopt_string(zmq.SUBSCRIBE, 'RejectOrder')
        self.order_manager.setsockopt_string(zmq.SUBSCRIBE, 'RejectCancelOrder')
        pass

    def startClient(self, processTickData):
        while True:
            msg = self.order_manager.recv_string()
            msg_type = msg.split('\t')[0]
            if msg_type == self.stock:
                processTickData(msg)
            else:
                comeback = msg.split('\t')[1]
                self.updateOrder(comeback, msg_type)

    def sendOrder(self, order):
        logger.info("order_client sendOrder start...")
        json_order = json.loads(order)
        self.order_client.send_string(order)
        msg = self.order_client.recv_string()
        if msg != '':
            tradeJson = json.loads(msg)
            if 'clOrderID' in tradeJson:                # 没懂
                clOrderID = tradeJson['clOrderID']
                _orders[clOrderID] = {}
                _orders[clOrderID]['tradeTime'] = json_order['tradetime']   #交易时间
                _orders[clOrderID]['reqType'] = json_order['reqType']   # 
                _orders[clOrderID]['orderSide'] = json_order['orderSide']   # 交易类型，买入或者卖出
                _orders[clOrderID]['securityID'] = json_order['securityID']     # 股票代码
                _orders[clOrderID]['orderQty'] = json_order['orderQty']     # 委托数量
                _orders[clOrderID]['price'] = json_order['price']           # 委托价格
                _orders[clOrderID]['orderType'] = json_order['orderType']   # 委托方式，对手方最优，限价等~
                _orders[clOrderID]['orderStatus'] = ''                      # 交易状态，可能等待交易所的写入，匹配
                _orders[clOrderID]['actVolume'] = 0     # 实际成交量
                _orders[clOrderID]['actPrice'] = 0      # 成交价格
                _orders[clOrderID]['cancelQty'] = 0         # 取消的量？
                _orders[clOrderID]['exeReportSeq'] = 0  # 序号
                self.sum_order_qty = self.sum_order_qty + int(json_order['orderQty']) # 以前的持仓+现在的买入
                logger.info("order_client sendOrder end...%s,%s" % (clOrderID, str(_orders[clOrderID])))
        # print(_orders)
        pass

    def cancelOrder(self, clOrderID):
        logger.info("order_client cancelOrder start...")
        result = {}
        result['reqType'] = 'cancelOrder'
        result['clOrderID'] = clOrderID
        print(result)
        logger.info(str(result))
        self.order_client.send_string(json.dumps(result))
        msg = self.order_client.recv_string()
        logger.info("order_client cancelOrder end...")
        pass

    def updateOrder(self, msg, msg_type):
        trade_json = json.loads(msg)
        clOrderID = trade_json['clOrderID']
        if msg_type == 'Trade':
            now_req = int(trade_json['exeReportSeq'])
            if now_req > _orders[clOrderID]['exeReportSeq']:
                print(trade_json)
                cancelQty = trade_json['canceledQty']
                cumQty = trade_json['cumQty']  # 实际成交量
                lastpx = trade_json['lastPx']
                _orders[clOrderID]['actVolume'] = int(cumQty)
                _orders[clOrderID]['actPrice'] = lastpx
                _orders[clOrderID]['orderStatus'] = trade_json['orderStatus']
                _orders[clOrderID]['cancelQty'] = cancelQty
                _orders[clOrderID]['exeReportSeq'] = now_req
                logger.info("update order...%s" % str(trade_json))
        elif msg_type == 'RejectOrder':
            self.sum_order_qty = self.sum_order_qty - _orders[clOrderID]['orderQty']
            self.sum_cancel_qty = self.sum_cancel_qty + _orders[clOrderID]['orderQty']
            _orders[clOrderID]['orderStatus'] = '9'
            logger.info("reject order...%s" % str(self.sum_cancel_qty))
        elif msg_type == 'RejectCancelOrder':
            cumQty = int(trade_json['cumQty'])
            if cumQty == int(_orders[clOrderID]['orderQty']):
                _orders[clOrderID]['orderStatus'] = '8'
            elif cumQty < int(_orders[clOrderID]['orderQty']) and cumQty > 0:
                _orders[clOrderID]['orderStatus'] = '7'
            else:
                print('wrong')
            logger.info("RejectCancelOrder...")
            pass

    def updateCancelOrder(self, clOrderID, status):
        _orders[clOrderID]['orderStatus'] = status

    def get_sum_order_qty(self):
        return self.sum_order_qty

    def get_sum_cancel_qty(self):
        return self.sum_cancel_qty

    def get_info(self):
        sum_comQty = 0
        sum_cancelQty = 0
        for clOrderID in _orders:
            sum_comQty = sum_comQty + _orders[clOrderID]['actVolume']
            sum_cancelQty = sum_cancelQty + _orders[clOrderID]['cancelQty']
        return sum_comQty, sum_cancelQty

    @staticmethod
    def getTradeSnapShot():
        global _orders
        return _orders
