# -*- coding: utf-8 -*-
import os
import zmq
import sys
import json
import numpy as np
import logging
import datetime

import uuid
import heapq
import pandas as pd

# from keras.models import load_model

dirs = 'logs'
if not os.path.exists(dirs):
    os.makedirs(dirs)

logging.basicConfig(filename='%s/info_order_client.log' % dirs, level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
_orders = {}
_books = []

class order_client:
    sum_order_qty = 0
    sum_cancel_qty = 0

    def __init__(self, xquantid, tickPath):
        self.xquantid = xquantid
        self.allTickData = pd.read_csv(tickPath, encoding='gb2312')
        print(self)
        self.allTickData.iloc[:, 54] = self.allTickData.iloc[:, 54] / 10000

        self.flist = self.read_fet_content()
        # print(self.flist)
        # self.find = {}
        # for i, k in enumerate(self.flist):
        #     self.find[k] = i
        pass

    def read_fet_content(self):
        fet_list = []
        with open('fet_cont_vwap.txt', 'r') as fin:
            for line in fin:
                fet_list.append(line.strip('\n'))
        return fet_list

    def startClient(self, processTickData):

        for i in range(self.allTickData.shape[0]):
            msg = [self.allTickData.iloc[i]["ticktime"]]
            for fea in self.flist:
                msg.append(self.allTickData.iloc[i][fea])

            tickInfo = [self.allTickData.iloc[i]["ticktime"],self.allTickData.iloc[i]["price"],
            self.allTickData.iloc[i+1]["price"],self.allTickData.iloc[i]["Buy1Price"],self.allTickData.iloc[i]["Buy1OrderQty"],
            self.allTickData.iloc[i]["Sell1Price"],self.allTickData.iloc[i]["Sell1OrderQty"]]
            self.matchDeal(tickInfo)
            status = processTickData(msg)

            if 'E' == status:
                print("exit by strtegy and _books has %d orders"%len(_books))
                break

    def matchDeal(self,msg):

        while _books:
            bestOrder = heapq.heappop(_books)
            print(bestOrder)
            clOrderID = bestOrder[-1]
            #TODO:def matchDealMethod(msg, clOrderID)
            actPrice, actVolume, orderStatu, aheadVol = self.matchDealMethod(msg, clOrderID)
            _orders[clOrderID]['orderStatus'] = orderStatu
            _orders[clOrderID]['actVolume'] = actVolume
            _orders[clOrderID]['actPrice'] = actPrice
            print("Match %s  %s  %s"%(clOrderID, bestOrder[2], bestOrder[0]*-1))
            #bestOrder = heapq.nsmallest(1, self._books)
        pass

    def matchDealMethod(self, msg, clOrderID):
        # msg[3] 买1价, msg[4]下一个买1价, msg[5]买1量, msg[6]下一个买1量, 
        # msg[7]成交量, msg[8]下一个成交量, 
        if  _orders[clOrderID]['price'] > msg[2]:
            act_price = msg[2]
            act_vol = _orders[clOrderID]['orderQty']
            order_status = '8'     #成功
        elif _orders[clOrderID]['price'] < msg[2]:
            act_price = 0
            act_vol = 0
            order_status = False
            ahead_vol = msg[5] - _orders[clOrderID]['orderQty']
            if msg[3] == msg[4]:        # 需要下一个买一价
                if msg[6] < msg[5]:
                    ahead_vol = ahead_vol - (msg[5] - msg[6])
        else:
            if msg[8] == msg[7]:
                act_price = 0
                act_vol = 0
                order_status = False
                ahead_vol = msg[5] - _orders[clOrderID]['orderQty']
                if msg[4] == msg[3] and msg[3] == _orders[clOrderID]['price']:
                    if msg[6] < msg[5]:
                        ahead_vol = ahead_vol - (msg[5] - msg[6])
            elif msg[8] > msg[7]:
                change = msg[8] - msg[7]
                if change <= ahead_vol:
                    act_price = 0
                    act_vol = 0
                    order_status = False
                    ahead_vol = ahead_vol - change
                else:
                    if change - ahead_vol >= _orders[clOrderID]['orderQty']:
                        act_price = _orders[clOrderID]['price']
                        act_vol = _orders[clOrderID]['orderQty']
                        order_status = '8'     #成功
                    else:
                        act_price = _orders[clOrderID]['price']
                        act_vol = change - ahead_vol
                        _orders[clOrderID]['orderQty'] = _orders[clOrderID]['orderQty'] - act_vol
                        order_status = '8'     #部分成功


            return act_price, act_vol, order_status,ahead_vol

    def sendOrder(self, order):
            logger.info("order_client sendOrder start...")
            clOrderID = str(uuid.uuid4())
            _orders[clOrderID] = {}
            _orders[clOrderID]['tradeTime'] = order['tradetime']
            _orders[clOrderID]['reqType'] = order['reqType']
            _orders[clOrderID]['orderSide'] = order['orderSide']
            _orders[clOrderID]['securityID'] = order['securityID']
            _orders[clOrderID]['orderQty'] = order['orderQty']
            _orders[clOrderID]['price'] = order['price']
            _orders[clOrderID]['orderType'] = order['orderType']
            _orders[clOrderID]['orderStatus'] = ''
            _orders[clOrderID]['actVolume'] = 0
            _orders[clOrderID]['actPrice'] = 0
            _orders[clOrderID]['cancelQty'] = 0
            _orders[clOrderID]['exeReportSeq'] = 0
            self.sum_order_qty = self.sum_order_qty + int(order['orderQty'])
            logger.info("order_client sendOrder end...%s,%s" % (clOrderID, str(_orders[clOrderID])))

            heapq.heappush(_books , (-1*order['price'] , order['tradetime'], order['orderQty'], clOrderID))
            # print(_orders)
            pass

    def cancelOrder(self, clOrderID):
            logger.info("order_client cancelOrder start...")
            #判断这个订单号对应的实际成交量和委托量之间的大小，5 部撤 6 已撤
            if _orders[clOrderID]['actVolume'] == 0:
                    _orders[clOrderID]['cancelQty'] = _orders[clOrderID]['orderQty']
                    _orders[clOrderID]['orderStatus'] = '6'
            else:
                    _orders[clOrderID]['cancelQty'] = _orders[clOrderID]['orderQty'] - _orders[clOrderID]['actVolume']
                    _orders[clOrderID]['orderStatus'] = '5'
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
                            cumQty = trade_json['cumQty']
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
            return _orders

    return _orders
