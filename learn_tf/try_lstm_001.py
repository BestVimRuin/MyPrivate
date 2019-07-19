import cx_Oracle
import pandas as pd
import talib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from  keras.models import Sequential
from  keras.layers import Dense
from  keras.layers import LSTM
from  keras.layers import Dropout
import matplotlib.pyplot as plt


conn = cx_Oracle.connect('factor_factory/htfactor123@168.61.13.175:1521/ost')
conn = cx_Oracle.connect('unified_factor/AVIQ_ff_79@168.61.2.4:1521/gbk')
sql = 'select s_info_windcode,S_DQ_CLOSE,S_DQ_HIGH,S_DQ_LOW,S_DQ_PRECLOSE,S_DQ_VOLUME,trade_dt from ' \
      'UNIFIED_FACTOR_ZX.AShareEODPrices t where s_info_windcode=\'601688.SH\' AND trade_dt between {} and {}'.format(20170117, 20190117)
df = pd.read_sql(sql, con=conn)
df.sort_values(by='TRADE_DT',ascending=True,inplace=True)
columns_list = ['S_DQ_CLOSE','S_DQ_HIGH','S_DQ_LOW','S_DQ_PRECLOSE','S_DQ_VOLUME']
MA7 =pd.DataFrame()
for i in columns_list:
    MA7[i] = talib.MA(df[i].values,timeperiod=7)
    
MA30 =pd.DataFrame()
for i in columns_list:
    MA30[i] = talib.MA(df[i].values,timeperiod=7)

sc = MinMaxScaler(feature_range=(0,1))
MA7.dropna(how='all',inplace=True)


MA7 = pd.DataFrame(sc.fit_transform(MA7))
qqq = MA7
MA7_train,MA7_test = MA7.iloc[:400,0:1],MA7.iloc[400:,0:1]
train7 = MA7_train
# train30 = sc.fit_transform(MA30)
x=[]
y=[]
for i in range(len(train7)-60):
    x.append(list(train7.iloc[i:i+60,0]))
    y.append(train7.iloc[i,0])

x,y = np.array(x),np.array(y)

x = np.reshape(x,(x.shape[0],x.shape[1],1))

regressor = Sequential()

regressor.add(LSTM(units=50,return_sequences=True,input_shape=(x.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True,))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True,))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss='mean_squared_error')
# regressor.compile(optimizer='adam',loss='mse')
regressor.fit(x,y,epochs=50,batch_size=32)

x_test=[]
y_test=[]
test7 = MA7_test
for i in range(len(test7)-60):
    x_test.append(list(test7.iloc[i:i+60,0]))
    y_test.append(test7.iloc[i,0])


x_test,y_test = np.array(x_test),np.array(y_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

y_predict = regressor.predict(x_test)



plt.plot(y_predict,color='red',label='预测值')
plt.plot(y_test,color='blue',label='真实值')
plt.show()
plt.close()









