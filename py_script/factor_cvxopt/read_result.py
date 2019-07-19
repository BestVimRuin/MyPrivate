import codecs
import json
import pandas as pd
import matplotlib.pyplot as plt


def client():
    with codecs.open('客户端返回.txt', 'r', 'utf-8') as f_result:
        f_lines = f_result.read()
        train = json.loads(f_lines)


def product():
    with codecs.open('产品侧真实数据.txt', 'r', 'utf-8') as f_result:
        f_lines = f_result.read()
        train = json.loads(f_lines)


def read_txt(file_name):
    with codecs.open(r'txt/style/{}.txt'.format(file_name), 'r', 'utf-8') as f_result:
        f_lines = f_result.read()
        result = json.loads(f_lines)
        result_df = pd.DataFrame(result)
        return result_df


txt_7 = read_txt('7')
txt_12 = read_txt('12')
txt_15 = read_txt('15')
txt_30 = read_txt('30')
txt_40 = read_txt('40')
txt_60 = read_txt('60')

plt.subplot(321)
width = 0.35  # the width of the bars: can also be len(x) sequence
x_ticks = txt_7.columns
# x_ticks = pd.to_datetime(txt_7.columns, format='%Y-%m-%d')
p1 = plt.bar(x_ticks, txt_7.iloc[0], width, color='green')
p2 = plt.bar(x_ticks, txt_7.iloc[1], width, color='yellow', bottom=txt_7.iloc[0])
p3 = plt.bar(x_ticks, txt_7.iloc[2], width, color='red', bottom=txt_7.iloc[1])
p4 = plt.bar(x_ticks, txt_7.iloc[3], width, color='black', bottom=txt_7.iloc[2])
p5 = plt.bar(x_ticks, txt_7.iloc[4], width, color='grey', bottom=txt_7.iloc[3])
p6 = plt.bar(x_ticks, txt_7.iloc[5], width, color='pink', bottom=txt_7.iloc[4])

plt.subplot(322)
width = 0.35  # the width of the bars: can also be len(x) sequence
x_ticks = txt_12.columns
y = txt_12
# x_ticks = pd.to_datetime(txt_7.columns, format='%Y-%m-%d')
p1 = plt.bar(x_ticks, txt_12.iloc[0], width, color='green')
p2 = plt.bar(x_ticks, txt_12.iloc[1], width, color='yellow', bottom=txt_12.iloc[0])
p3 = plt.bar(x_ticks, txt_12.iloc[2], width, color='red', bottom=txt_12.iloc[1])
p4 = plt.bar(x_ticks, txt_12.iloc[3], width, color='black', bottom=txt_12.iloc[2])
p5 = plt.bar(x_ticks, txt_12.iloc[4], width, color='grey', bottom=txt_12.iloc[3])
p6 = plt.bar(x_ticks, txt_12.iloc[5], width, color='pink', bottom=txt_12.iloc[4])

plt.subplot(323)
width = 0.35  # the width of the bars: can also be len(x) sequence
x_ticks = txt_15.columns
y = txt_15
# x_ticks = pd.to_datetime(txt_7.columns, format='%Y-%m-%d')
p1 = plt.bar(x_ticks, y.iloc[0], width, color='green')
p2 = plt.bar(x_ticks, y.iloc[1], width, color='yellow', bottom=y.iloc[0])
p3 = plt.bar(x_ticks, y.iloc[2], width, color='red', bottom=y.iloc[1])
p4 = plt.bar(x_ticks, y.iloc[3], width, color='black', bottom=y.iloc[2])
p5 = plt.bar(x_ticks, y.iloc[4], width, color='grey', bottom=y.iloc[3])
p6 = plt.bar(x_ticks, y.iloc[5], width, color='pink', bottom=y.iloc[4])


plt.subplot(324)
width = 0.35  # the width of the bars: can also be len(x) sequence
x_ticks = txt_30.columns
y = txt_30
# x_ticks = pd.to_datetime(txt_7.columns, format='%Y-%m-%d')
p1 = plt.bar(x_ticks, y.iloc[0], width, color='green')
p2 = plt.bar(x_ticks, y.iloc[1], width, color='yellow', bottom=y.iloc[0])
p3 = plt.bar(x_ticks, y.iloc[2], width, color='red', bottom=y.iloc[1])
p4 = plt.bar(x_ticks, y.iloc[3], width, color='black', bottom=y.iloc[2])
p5 = plt.bar(x_ticks, y.iloc[4], width, color='grey', bottom=y.iloc[3])
p6 = plt.bar(x_ticks, y.iloc[5], width, color='pink', bottom=y.iloc[4])

plt.subplot(325)
width = 0.35  # the width of the bars: can also be len(x) sequence
x_ticks = txt_40.columns
y = txt_40
# x_ticks = pd.to_datetime(txt_7.columns, format='%Y-%m-%d')
p1 = plt.bar(x_ticks, y.iloc[0], width, color='green')
p2 = plt.bar(x_ticks, y.iloc[1], width, color='yellow', bottom=y.iloc[0])
p3 = plt.bar(x_ticks, y.iloc[2], width, color='red', bottom=y.iloc[1])
p4 = plt.bar(x_ticks, y.iloc[3], width, color='black', bottom=y.iloc[2])
p5 = plt.bar(x_ticks, y.iloc[4], width, color='grey', bottom=y.iloc[3])
p6 = plt.bar(x_ticks, y.iloc[5], width, color='pink', bottom=y.iloc[4])


plt.subplot(326)
width = 0.35  # the width of the bars: can also be len(x) sequence
x_ticks = txt_60.columns
y = txt_60
# x_ticks = pd.to_datetime(txt_7.columns, format='%Y-%m-%d')
p1 = plt.bar(x_ticks, y.iloc[0], width, color='green')
p2 = plt.bar(x_ticks, y.iloc[1], width, color='yellow', bottom=y.iloc[0])
p3 = plt.bar(x_ticks, y.iloc[2], width, color='red', bottom=y.iloc[1])
p4 = plt.bar(x_ticks, y.iloc[3], width, color='black', bottom=y.iloc[2])
p5 = plt.bar(x_ticks, y.iloc[4], width, color='grey', bottom=y.iloc[3])
p6 = plt.bar(x_ticks, y.iloc[5], width, color='pink', bottom=y.iloc[4])
plt.show()
