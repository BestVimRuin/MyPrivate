# -*- coding: utf-8 -*-
import sys
from importlib import reload
import pandas as pd
from dao import htsc_oracle as om


print(sys.getdefaultencoding())


df = pd.read_excel('华泰研报因子分类.csv')
# format = lambda x: '%.2f' % x
format = lambda x: x.strip()
df.applymap(str)
df.applymap(format)
info_list = df.values.tolist()
flag = om.info_insert(info_list)
print(flag)









