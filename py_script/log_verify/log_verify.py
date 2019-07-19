~# -*- coding: utf-8 -*-
def log_check(): 
    import re
    import os
    import json
    import logging
    import time
    import collections

    start_time = time.time()
    dirs = 'Res'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    logging.basicConfig(filename='%s/log_verify.log' % dirs, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 处理最高价,最低价
    def operate_price(pattern, line):
        operate_price = re.findall(pattern, line)
        if operate_price != []:
            operate_price_re =re.sub('\'','\"',operate_price[0])
            operate_price_re = operate_price_re.split(":")[-1]
        else :
            return False
        return operate_price_re

    status_turn = {}
    dir = '/home/lss/ESA-AUTOTEST/Strategy/logs/'
    dir_files = os.listdir(dir)

    for i in dir_files:
        num = 0
        sell_flag = []
        total_orderqty_flag = []
        tradetime_flag = []
        orderStatus_flag = []
        cumQty_flag = []
        orderStatus_cumQty_flag = []
        orderStatus_canceledQty_flag = []
        max_min_flag = []
        deadline = False
        remain_verify = 0
        with open("{}/{}".format(dir, i), 'r') as f:
            logger.info('name : {}'.format(i))
            for line in f.readlines():
                temp_l = []
                # 选取每行的{}字典中的内容
                line_re = re.findall(r'{.*}', line)
                # deadline是过期结束,正常结束时total_remain = 0
                DeadLine = re.findall(r'deadline', line)
                TotalRemain = re.findall(r'total.remain.*', line)
                
                
                # 每行的需要获取的值
                MaxPrice = operate_price(r'MaxPrice.*', line)
                MinPrice = operate_price(r'MinPrice.*', line)
                LimitPrice = operate_price(r'limit.price.*', line)
                TotalOrderqty = operate_price(r'total.orderqty.*', line)
                LastPx = operate_price(r'lastpx.*', line) 
                price_list = re.findall(r'\{\'\d.*\d\}', line)            

                if MaxPrice:
                    max_price = float(MaxPrice)
                if MinPrice:
                    min_price = float(MinPrice)
                if LimitPrice:
                    limit_price = float(LimitPrice)
                if TotalOrderqty:
                    total_orderqty = float(TotalOrderqty)
                if LastPx:
                    last_px = float(LastPx)
                if TotalRemain:
                    TotalRemainNum = re.findall(r'\d+', TotalRemain[0])
                    total_remain = TotalRemainNum[0]
                if DeadLine:
                    deadline = DeadLine[0]
                if price_list:
                    
                    price = price_list[0]
                    price=re.sub('\'', '\"', price)
                    price_dict = json.loads(price)
                    for j in price_dict.keys():
                        temp_l.append(j)
                    temp_l.sort()         # sort的参数reverse = False 升序（默认）
                        
                
                # 当做日志里面的行号
                num += 1
                # 错误类型
                kind = ''
                # 字典中的内容不为空,
                # if 0:
                if line_re != []:
                
                    test=re.sub('\'', '\"', line_re[0])
                    # my_dict_json就是字典中的内容
                    my_dict_json = json.loads(test)
                    if 'price' in my_dict_json.keys():
                        price_float = float(my_dict_json['price'])
                    if temp_l != []:
                        temp_l_min = float(temp_l[0])
                        temp_l_max = float(temp_l[2])
                        
                        if 'price' in my_dict_json.keys():
                            price_float = float(my_dict_json['price'])
                            if(price_float < temp_l_min or  price_float > temp_l_max):
                                # error18
                                sell_flag.append(0)
                            else:
                                sell_flag.append(1)
                    # 这个可以判断服务端 and 客户端的的委托量
                    if 'orderQty' in my_dict_json.keys() and 'clOrderID' in my_dict_json.keys():
                        if float(my_dict_json['orderQty']) > float(total_orderqty)*0.1:
                            # error10
                            total_orderqty_flag.append(0)
                        else:
                            total_orderqty_flag.append(1)
                    if 'tradetime' in my_dict_json.keys():
                            if float(my_dict_json['tradetime']) > 145500000 and float(my_dict_json['orderQty']) > 4000:
                                # error11
                                tradetime_flag.append(0)
                            else:
                                tradetime_flag.append(1)

                    # 判断服务端
                    if 'reqType' not in my_dict_json.keys() and 'clOrderID' in my_dict_json.keys():
                        if float(my_dict_json['orderStatus']) == 2 and float(my_dict_json['cumQty']) != 0:
                            # error12
                            orderStatus_flag.append(0)
                        else:
                            orderStatus_flag.append(1)
                        if float(my_dict_json['orderStatus']) == 7 and \
                                (float(my_dict_json['cumQty']) <= 0 or float(my_dict_json['cumQty']) >= float(my_dict_json['orderQty'])) :
                            # error13
                            cumQty_flag.append(0)
                        else:
                            cumQty_flag.append(1)
                        if float(my_dict_json['orderStatus']) == 8 and float(my_dict_json['cumQty']) != float(my_dict_json['orderQty']) :
                            # error14
                            orderStatus_cumQty_flag.append(0)
                        else:
                            orderStatus_cumQty_flag.append(1)
                        if float(my_dict_json['orderStatus']) == 5 and  status_turn[my_dict_json['clOrderID']][0] ==7 and\
                                float(my_dict_json['canceledQty']) != (status_turn[my_dict_json['clOrderID']][1] - status_turn[my_dict_json['clOrderID']][2]):
                            # error15
                            orderStatus_canceledQty_flag.append(0)
                        else:
                            orderStatus_canceledQty_flag.append(1)
                        if price_float > max_price or price_float < min_price or price_float <= 0:
                            # error16
                            max_min_flag.append(0)
                        else:
                            max_min_flag.append(1)

                        # 最后每次都更新状态的转变,用于部成到部撤的时候,存的是同一个订单号的上一个orderStatus
                        status_turn[my_dict_json['clOrderID']] = [float(my_dict_json['orderStatus']), float(my_dict_json['orderQty']), float(my_dict_json['cumQty'])]

            sell_flag = collections.Counter(sell_flag)
            total_orderqty_flag = collections.Counter(total_orderqty_flag)
            tradetime_flag = collections.Counter(tradetime_flag)
            orderStatus_flag = collections.Counter(orderStatus_flag)
            cumQty_flag = collections.Counter(cumQty_flag)
            orderStatus_cumQty_flag = collections.Counter(orderStatus_cumQty_flag)
            orderStatus_canceledQty_flag = collections.Counter(orderStatus_canceledQty_flag)
            max_min_flag = collections.Counter(max_min_flag)

            if len(total_orderqty_flag) != 1:
                logger.error('10 : error')
            else:
                logger.info('10 : success')

            if len(tradetime_flag) != 1:
                logger.error('11 : error')
            else:
                logger.info('11 : success')

            if len(orderStatus_flag) != 1:
                logger.error('12 : error')
            else:
                logger.info('12 : success')

            if len(cumQty_flag) != 1:
                logger.error('13 : error')
            else:
                logger.info('13 : success')

            if len(orderStatus_cumQty_flag) != 1:
                logger.error('14 : error')
            else:
                logger.info('14 : success')

            if len(orderStatus_canceledQty_flag) != 1:
                logger.error('15 : error')
            else:
                logger.info('15 : success')

            if len(max_min_flag) != 1:
                logger.error('16 : error')
            else:
                logger.info('16 : success')

            if float(total_remain) != 0:
                kind = '17'
                logger.warning('%s : warning, line : %s, name : %s' % (kind, num, i))
            else:
                kind = '17'
                logger.info('%s : success' % (kind))
    end_time = time.time()  
    print(end_time - start_time)    


if __name__ == '__main__':
    log_check()
