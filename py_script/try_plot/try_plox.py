# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 处理输入,分别返回
# parameter_str = input("Please input your order:filename,datetime,nums,show_xaxis(0 or 1)"
      # "such as 20180801 2000 1"
      # "input:")
      
# 输入分别为文件名,初始日期,结束日期,是否显示y=0的直线(1为显示)
parameter_str = '300699_DRWA_basic.csv,20170927,20170930,1'
parameter_list = parameter_str.split(',')
filename = parameter_list[0]
start_date = int(parameter_list[1])
end_date = int(parameter_list[2])
show_xaxis = int(parameter_list[3])

# 定义当前文件所在目录,用于遍历输入的文件
# dir = os.path.abspath('.')    #返回绝对路径
# file_dir = "{}\\data".format(dir)
# files = os.listdir(file_dir)
dirs = "pics\\{}".format(parameter_list[1])
if not os.path.exists(dirs):
    os.makedirs(dirs)
    
# 直接读取所选文件
df = pd.read_csv("data\\{}".format(filename))

# 对每列的数据进行数字化,避免出现str
for i in range(df.shape[1]):
    df[df.columns[i]] = pd.to_numeric(df[df.columns[i]], errors='coerce')
df.drop(['label'], axis = 1, inplace=True)  # 删除第2列
df_filter = df[np.array(df['Date']>=start_date) & np.array(df['Date'] < end_date)]
df_filter.index = range(df_filter.shape[0])
for column in df_filter.columns:
    # df_rand = df[column][rand].sort_index()     # 每个Series数据的随机2000行
    fig = plt.figure()                   # 定义dpi或者画布的一些其他信息
    plt.plot(df_filter.index, df_filter[column])
    # show_xaxis为1时,显示y=0的直线
    if show_xaxis:
        l = plt.axhline(y=0, color='b')
    plt.title(column)
    plt.savefig("{}\\{}.png".format(dirs, column))
    plt.close()                                   # 记得关闭每个画布,要不就会画到一起

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


