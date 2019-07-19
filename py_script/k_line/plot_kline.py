# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt



start_date = 20170101
end_date = 20180601
percent = 0.95
filename_dir = 'D:\\\code\\htsc\\k_line\\date_code\\SH\\20180102'
filename_list = os.listdir(filename_dir)
stock_filename_dir = 'D:\\\code\\htsc\\k_line\\day_line\\'
dir = 'D:\\\code\\htsc\\k_line\\date_code\\SH\\'
dir_files = os.listdir(dir)
result_dir = 'D:\\\code\\htsc\\k_line\\result\\'
err_num = 0

for filename in filename_list:
    stock_filename = stock_filename_dir + filename
    df_stock = pd.read_csv(stock_filename, encoding='gb2312')
    df_stock.drop([0], inplace=True)
    # 对每列的数据进行数字化,避免出现str
    total_ratio = pd.DataFrame()
    for i in range(df_stock.shape[1]):
        df_stock[df_stock.columns[i]] = pd.to_numeric(df_stock[df_stock.columns[i]], errors='coerce')
    for i in dir_files:
        if int(i) >= start_date & int(i) <= end_date:
            # if not os.path.exists('{}\\data\\{}'.format(dir, i)):
            #     os.makedirs('data\\{}'.format(i))
            my_dict = {}
            try:
                df_day = pd.read_csv(dir + i + '\\' + filename, encoding='gb2312')
            except Exception as e:
                err_num += 1
                continue
            
            df_day.drop([0], inplace=True)
            df_day['收盘价'] = pd.to_numeric(df_day['收盘价'], errors='coerce')
            per_close = df_stock[df_stock['日期']==int(i)]
            print(i)
            try:
                ratio = df_day['收盘价']/float(per_close['收盘价']) -1
            except Exception as e:
                err_num += 1
                continue
            ratio.index = df_day['时间']
            # 对实时的收盘价/最后的收盘价比率,取绝对值
            # ratio = ratio.abs()
            # m = ratio.shape[0]
            # for j in range(m):
            #     nums = (m-j) * percent
            #     nums = int(round(nums, 0))
            #     df_sort = ratio.iloc[j:].sort_values()
            #     # 取到符合条件的最大波动范围
            #     df_nums = df_sort[:nums][-1:]
            #     value = np.array(df_nums)[0]
            #     my_dict[ratio.index[j]] = value
            # my_data = pd.Series(my_dict)
            total_ratio = pd.concat([total_ratio, ratio], axis=1, sort=False)
    total_ratio.index = pd.to_numeric(total_ratio.index, errors='coerce')
    total_ratio = total_ratio.sort_index()
    total_ratio.dropna(axis=1, how='all', inplace=True)
    # total_ratio.to_csv('total_ratio.csv')
    ratio_std= total_ratio.std(axis=1)
    # sss = list(ratio_std.index) 
    df1 = pd.DataFrame(list(ratio_std.index))
    df2 = pd.DataFrame(ratio_std.values)
    df = pd.concat([df1, df2],axis=1)
    df.columns = ['MDTime','Std']
    df['MDTime'] = df['MDTime']/100000
    df.to_csv(result_dir + filename,index=False)

print(err_num)


    
    
        
        
    
