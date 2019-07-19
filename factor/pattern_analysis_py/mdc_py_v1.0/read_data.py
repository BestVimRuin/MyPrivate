#coding=utf-8
import os
import cx_Oracle 
import re
import time
#from newsearch import *
start_time=time.time()
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import sys
import logging
# 设置工作目录 #
workpath = "D:\\maticServer\\factor_matic_server\\mdc_pattern_analysis\\"
sys.path.append(workpath)

# 将INFO级别或更高的日志信息打印到log文件
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%d %b %Y %H:%M:%S',
                filename= workpath+'stock_predict_task.log',
                filemode='w+')

#获取当前日期
now_date = time.strftime('%Y%m%d', time.localtime(time.time()))
now_date = '20180524'
logging.info("Now do the stock predict task ,date:%s" %(now_date))

#从数据库中获取后五天交易日
conn=cx_Oracle.connect('ZJ013_INFO/ZJ013_INFO@168.9.2.43:1521/qdb04')
cursor = conn.cursor()
sql='SELECT * FROM (SELECT DISTINCT TRADE_DAYS FROM ZJ013_DATA.ASHARECALENDAR WHERE TRADE_DAYS>%s ORDER BY TRADE_DAYS) WHERE ROWNUM<=5'%now_date

cursor.execute(sql)
date = cursor.fetchall()
cursor.close()
conn.close()
date=[i[0] for i in date]


#读取收盘价
f=open("D:\\maticServer\\mdc_pattern_analysis\\marketValue.txt","r")
data=f.read().split("\n")
f.close()
#去0处理
new_data=[]
for line in data:
	line = line.split(":")
	# print(line)
	price = line[1].split(",")
	price = [float(x) for x in price]

	new_price = []
	if price[0]!=0:
		for i in range(len(price)):
			if price[i]!=0:
				new_price.append(price[i])
		if len(new_price)>=20:
			new_price.append(line[0])
			new_data.append(new_price)
		
# print(new_data)
info=[]
corr=[]
for row in new_data:
	one_info = []

	price = row[:20][::-1]

	result = compare(price)
	Closing_price = float(price[-1])
	# corr.append(result["corr"])

	one_info.append(now_date)			#数据日期
	one_info.append(row[-1])			#基金代码
	one_info.append(Closing_price)		#收盘价

	one_info.append(date[int(result["highpoint"])])					#预测最大值日期
	one_info.append(Closing_price+float(result["maxampTop"]))		#最大值最大值
	one_info.append(Closing_price+float(result["maxampButton"]))	#最大值最小值
	one_info.append(date[int(result["lowpoint"])])					#预测最小值日期
	one_info.append(Closing_price+float(result["minampTop"]))		#最小值最大值
	one_info.append(Closing_price+float(result["minampButton"]))	#最小值最小值
	if result["upChance"]<0.45:
		s="下跌"
		one_info.append(s)						#预测趋势
		one_info.append(1-result["upChance"])	#预测概率
	elif result["upChance"]>0.55:
		s="上涨"
		one_info.append(s)						#预测趋势
		one_info.append(result["upChance"])		#预测概率
	else:
		s="持平"
		one_info.append(s)						#预测趋势
		one_info.append(result["upChance"])		#预测概率
	info.append(one_info)
						
	logging.info(one_info)

logging.info(info)
# print(corr)


# 删除数据库中当日的数据
conn=cx_Oracle.connect('ZJ013_INFO/ZJ013_INFO@168.9.2.43:1521/qdb04')
cursor = conn.cursor()
sql='DELETE FROM M_STOCK_PREDICT_INFO WHERE DATA_DATE=%s'%now_date
cursor.execute(sql)
conn.commit()
cursor.close()
conn.close()

#将匹配结果插入数据库
conn=cx_Oracle.connect('ZJ013_INFO/ZJ013_INFO@168.9.2.43:1521/qdb04')
cursor = conn.cursor()
sql = "INSERT INTO %s VALUES(:DATA_DATE,:STOCK_CODE,:STOCK_CLOSE,:PRED_MAX_DATE,:MAX_MAX_VALUE,:MAX_MIN_VALUE,:PRED_MIN_DATE,:MIN_MAX_VALUE,:MIN_MIN_VALUE,:PRED_DIRECTION,:PRED_PROB)"%"M_STOCK_PREDICT_INFO"

# 日期，股票代码，收盘价（该日期），预测后续最大值日期，最大最大值，最大最小值，预测后续最小值日期  最小最大值 最小最小值
cursor.executemany(sql, info)
cursor.execute('commit')  
conn.commit()
cursor.close()
conn.close()


stop_time=time.time()
logging.info("Task is done ,time consumes: %d"%(stop_time-start_time))

