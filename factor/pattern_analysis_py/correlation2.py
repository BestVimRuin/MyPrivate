# -*- coding: utf-8 -*-
## 2.1 纯粹物理意义相似定义(看涨跌幅变化趋势)
# 对原始数据进行标准化处理(只考虑涨跌幅——收益率标准化)：
from __future__ import division
import numpy as np

def stc(x): # 为了保证幅度的影响(1,2,3与10,20,30相同，而与11,12,13不同，因为幅度不同)
    ave=np.mean(x)
    if ave==0:
        stc=x-ave
    else:
        stc=(x-ave)/ave
    return stc
### (2).相似度计算函数——基于欧式距离比值
def OD_dist(v1,v2):
    d=np.sqrt(np.sum((v1-v2)**2))/(np.sqrt(np.sum((np.abs(v1)+np.abs(v2))**2)))
    d=1-d
    return d
# 局部相似度计算函数
## 包含涨跌幅不一致，但变化趋势一致的相似定义
### （1）OD
### 数据标准化
def stt(x,range0=1): # 均值，range处理,对close处理
    ave=np.mean(x)
    if(ave==0):
        temp=x-ave
    else:
        temp=(x-ave)/ave
#    range0=np.max(temp)-np.min(temp)
    if range0==0:
        y=temp-temp
    else:
        y=temp/range0
    return y

def stt1(x): # 减均值除以均值处理
    ave=np.mean(x)
    if(ave==0):
        temp=x-ave
    else:
        temp=(x-ave)/ave
    return temp
# range0取所有片段range的9分位点的数值，这样就去除了振幅不一样的影响
def stt2(temp,range0):# 除以range处理
#    range0=np.max(temp)-np.min(temp)
    if range0==0:
        stt=temp-temp
    else:
        stt=temp/range0
    return stt    

###基于欧式距离变周期
def Multiscale_tren_d(vv1,vv2):
    ## 先对原始距离进行标准化处理
#     vv1=stt2(v0)
#     vv2=stt(v)
    ## 局部处理
    N=len(vv1)
    NN=N
    d=np.array([])
    i=0
    ## 从1倍,1/2倍,1/4倍,1/8进行相似比对直至最后2点
    while(NN>2):
        v1=vv1[(N-NN):N] # 变周期如何搞定索引不为整数
        v2=vv2[(N-NN):N]
        d=np.insert(d,i,OD_dist(v1, v2))
        NN=int(np.floor(NN/2))
        i=i+1
    ## 计算最近两点的相似度
    closedist=OD_dist(vv1[(N-2):N], vv2[(N-2):N])
    d=np.insert(d,i,closedist)
    dis=np.mean(d)## 相似平均值：作为最终相似度
    ##阈值设定：
    num=np.sum(d[d<0.6])
    if((dis>=0.5) and (num<=len(d)/2) and (d[0]>(0.65))):
        judg=True
    else:
        judg=False
    q=np.repeat(1/len(d), len(d))
    q[0]=2*q[0]
    q[d[1:len(d)].argmin()+1]=0
    simi_d=np.sum(q*d)
    result={"judg":judg,"simi_d":simi_d,"d":[d]}
    return result

if __name__=="__main__":
#     v0=np.array([1.92081062 , 1.63168926 , 1.3425679  , 1.05344654 , 0.76432519  ,0.51866921,
#  0.40340939 , 0.28814956 , 0.17288974 , 0.05762991, -0.05762991 ,-0.17288974,
# -0.28814956 ,-0.40340939, -0.51866921,-0.76432519 ,-1.05344654 ,-1.3425679,
# -1.63168926 ,-1.92081062])
    v0=np.array([-1.92081062, -1.63168926, -1.3425679 , -1.05344654 ,-0.76432519 ,-0.51866921,\
-0.40340939 ,-0.28814956 ,-0.17288974 ,-0.05762991,  0.05762991  ,0.17288974,\
 0.28814956,  0.40340939,  0.51866921 , 0.76432519 , 1.05344654 , 1.3425679,\
 1.63168926 , 1.92081062])
    v=np.array([81265,79859,78488,77632,77460,77460,76946,71839,71873,73450,73587,76192,73279,77734,79105,76055,75575,75095,74547,74376])
#     v0=[2,1,-1,-2]
    # v0=[-2,-1,1,2]
    # v=[1,2,3,4]
    Multiscale_tren_d(v0,v)
