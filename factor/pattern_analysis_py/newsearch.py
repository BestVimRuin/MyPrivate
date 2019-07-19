# -*- coding: utf-8 -*-

"""

根据输入的股票价格片段，查询与它相似度高的形态。
并以字典的格式返回形态的信息。

"""

from __future__ import division
# from model import Shape,Zhibiao,Stat,Fragment,Conver,session,New
import json
import os
import sys
import correlation2
import numpy as np
import time
import cx_Oracle

def getShape():

    """

    获取所有形态的形态号及具体值

    :返回:
    - **[name, values](list)** : 返回包含形态信息的列表
    """
    # Query = session.query(Shape)
    conn=cx_Oracle.connect('ZJ013_INFO/ZJ013_INFO@168.9.2.43:1521/qdb04')
    cursor = conn.cursor()
    sql = "SELECT num,value FROM shape"
    cursor.execute(sql) 
    date = cursor.fetchall() 

    cursor.close()
    name = []
    values = []
    for shape in date:
        name.append(shape[0])
        values.append(str(shape[1]))
    # for shape in Query:
    #     name.append(shape.num)
    #     values.append(shape.value)

    return [name, values]


def transtype(x):

    """
    将传入的列表中的每个元素转化成float类型

    :参数：
    - **x(list, None)** : 由字符串组成的列表

    :返回:
    - **y(np.array)** : 返回一个一维数组

    """

    if x is None:
        y = None
    else:
        y = [np.float32(i) for i in x]
        y = np.array(y)
    return y


def scale(x,range0=1):

    """
    标准化函数

    :参数：
    - **x(list, None)** : 列表或数组
    - **range0, 1** : 

    :返回:
    - **y(list)** : 返回标准化后的序列
    """

    if x is None:
        y = None
    else:
        avg = np.mean(x)
        if  avg == 0:
            temp = x - avg
        else:
            temp = (x- avg)/avg
        if range0 == 0:
            y = temp - temp
        else:
            y = temp/range0
        y = y.tolist()
    return y





def compare(lst):

    """

    输入的股票价格与形态库中的形态进行比较，得到相似度最高的形态的形态号。
    根据这个形态号，到数据库中查询相关信息。

    :参数：
    - **lst(str, None)** : 输入的股票价格字符串，如：123,123,123,123,123....

    :返回:
    - **best_shape(dict)** : 以字典格式返回形态信息
    """

    shapes = getShape()
    v1 = np.array(scale(lst))
    temp = 0
    for i in range(len(shapes[1])):
        v2 = transtype(shapes[1][i].split('|'))
        c = correlation2.Multiscale_tren_d(v1,v2)['simi_d']
        if c >= temp:
            temp = c
            num = shapes[0][i]
    best_shape = searchShape(num)
    best_shape['corr'] = temp
    return best_shape

def searchShape(num):

    """

    根据传入的形态号，到数据库查询相关信息

    :参数：
    - **num(int, None)** : 形态号

    :返回:
    - **item(dict)** : 以字典格式返回相关信息
    """
    conn=cx_Oracle.connect('ZJ013_INFO/ZJ013_INFO@168.9.2.43:1521/qdb04')
    cursor = conn.cursor()
    sql = "select * from new where sid = (select id from shape where num = %s)"%(num)
    cursor.execute(sql) 
    data = cursor.fetchall()[0]
#    cursor = conn.cursor()
#    sql = "SELECT id FROM shape where num=%s"%(num)
#    cursor.execute(sql) 
#    shape_id = cursor.fetchall()[0]
##    print(shape_id)
#    sql = "SELECT * FROM new where sid=%s"%(shape_id)
#    cursor.execute(sql) 
#    data = cursor.fetchall()[0]
    cursor.close()

    item = {}
    # Query1 = session.query(Shape).filter(Shape.num == num).first()
    # print(Query1.num)
    # print(Query1.id)
    # Query2 = session.query(New).filter(New.sid == Query1.id).first()
    # print(Query2.sid)
    # Query2 = session.query(New).filter(New.num == num).first()
    item = {}
    item["upChance"] = data[3]
    item["highpoint"] = data[4]
    item["lowpoint"] = data[5]
    item["maxampButton"] = data[6]
    item["maxampTop"] = data[7]
    item["minampButton"] = data[8]
    item["minampTop"] = data[9]
    item["num"] = data[2]
    return item

if __name__ == '__main__':

    # 输入的股票价格片段
    s = sys.argv[1]
    # s = '59107-59819-60175-58751-60294-60531-61421-62430-63083-63795-62845-62489-62430-62964-64685-65635-66109-67059-69017-69077'
    lst = s.split(',')
    lst = [float(i) for i in lst]
    # 获取相似度最高的形态的信息
    result = compare(lst)
    # 编码成json格式
    result = json.dumps(result)
    # f = str(int(time.time()-1))
    # fp = open(f+'.txt', 'w')
    # fp.write(result)
    # fp.close()
    print(result)


    num =101
    Query1 = session.query(Shape).filter(Shape.num == num).first()
    Query2 = session.query(New).filter(New.sid == Query1.id).first()
    print(Query2.num,Query2.upChance)