import pandas as pd
import numba as np
import matplotlib.pyplot as plt
from dateutil.parser import parse

df = pd.read_csv('zl_group_cum_return.csv',encoding='gb2312')
code_300 = df[df['GROUP_ID'] == '000300.SH']
code_300 = code_300.sort_values(['HISTORY_DATE'])
fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#ax1.get_xaxis().get_major_formatter().set_useOffset(False)
#ax1.xaxis.set_major_formatter()
#ax1 = fig.add_subplot(1,1,1)
#ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
#plt.xticks(pd.date_range('2018-03-14','2018-09-20'),rotation=90)
# for i in range(1,17):
#     temp = df[df['GROUP_ID'] == (str(i)+'.0')]
#     temp = temp.sort_values(['HISTORY_DATE'])
#     temp['HISTORY_DATE'] = temp['HISTORY_DATE'].map(str)
#     plt.plot(temp['HISTORY_DATE'], temp['CUMULATIVE_RETURN']) 

code_300['HISTORY_DATE'] = code_300['HISTORY_DATE'].map(str)
code_300['HISTORY_DATE'] = pd.to_datetime(code_300['HISTORY_DATE'], format='%Y-%m-%d')
plt.plot(code_300['HISTORY_DATE'], code_300['CUMULATIVE_RETURN'])   
plt.show()
plt.close()                                   # 记得关闭每个画布,要不就会画到一起
plt.show()
