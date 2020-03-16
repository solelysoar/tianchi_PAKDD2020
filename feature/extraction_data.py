import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
import math
from lightgbm.sklearn import LGBMClassifier
from collections import Counter  
import time
from scipy.stats import kurtosis,iqr
from scipy import ptp
from tqdm import tqdm
import datetime
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss,f1_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import warnings
import os
import joblib
import pickle

cleaned=False
if not cleaned:
    print('fda')

if not cleaned:
    test = pd.read_csv('./data/disk_sample_smart_log_test_a.csv')
    tag = pd.read_csv('./data/disk_sample_fault_tag.csv')
    test['dt'] = test['dt'].apply(lambda x:''.join(str(x)[0:4] +'-'+ str(x)[4:6]  +'-'+ str(x)[6:]))      
    test['dt'] = pd.to_datetime(test['dt'])
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])

    ###tag表里面有的硬盘同一天发生几种故障
    tag['tag'] = tag['tag'].astype(str)
    tag = tag.groupby(['serial_number','fault_time','model'])['tag'].apply(lambda x :'|'.join(x)).reset_index()

    test = test.sort_values(['serial_number','dt'])
    test = test.drop_duplicates().reset_index(drop=True)

    ###去掉全为空值和nunique为1的特征
    drop_list =[]
    for i in tqdm([col for col in test.columns if col not in ['manufacturer','model']]):
        if (test[i].nunique() == 1)&(test[i].isnull().sum() == 0):
            drop_list.append(i)

    df= pd.DataFrame()
    df['fea'] = test.isnull().sum().index
    df['isnull_sum'] = test.isnull().sum().values
    fea_list = list(set(df.loc[df.isnull_sum != test.shape[0]]['fea']) - set(drop_list))

    test = test[fea_list]
    test.to_hdf('data/test.h5','1.0')
else:
    fea_list=['smart_7raw',
 'smart_198raw',
 'smart_12_normalized',
 'smart_241_normalized',
 'smart_192_normalized',
 'serial_number',
 'smart_5_normalized',
 'model',
 'smart_12raw',
 'smart_189raw',
 'smart_1_normalized',
 'smart_242raw',
 'smart_4_normalized',
 'smart_7_normalized',
 'smart_195raw',
 'smart_197_normalized',
 'smart_194raw',
 'smart_187_normalized',
 'smart_190_normalized',
 'smart_184_normalized',
 'smart_9raw',
 'smart_240raw',
 'smart_4raw',
 'smart_198_normalized',
 'smart_9_normalized',
 'smart_242_normalized',
 'smart_189_normalized',
 'smart_10raw',
 'smart_3_normalized',
 'manufacturer',
 'smart_241raw',
 'smart_5raw',
 'smart_193_normalized',
 'smart_240_normalized',
 'smart_199_normalized',
 'smart_188raw',
 'dt',
 'smart_199raw',
 'smart_195_normalized',
 'smart_1raw',
 'smart_184raw',
 'smart_187raw',
 'smart_10_normalized',
 'smart_192raw',
 'smart_193raw',
 'smart_194_normalized',
 'smart_197raw',
 'smart_190raw',
 'smart_188_normalized']
    test = pd.read_hdf('data/test.h5','1.0')

gc.collect()

if cleaned:

    ###去掉无用特征后给每个样本打上label，下次直接读取
    ##这里只处理了18年456月份的数据，其他同理，但是18年7月份的数据因为tag的时间也只到了七月份不好直接打label
    def get_label(df):
        df = df[fea_list]
        df['dt'] = df['dt'].apply(lambda x:''.join(str(x)[0:4] +'-'+ str(x)[4:6]  +'-'+ str(x)[6:]))      
        df['dt'] = pd.to_datetime(df['dt'])    

        df = df.merge(tag[['serial_number','model','fault_time']],how = 'left',on =['serial_number','model'])
        df['diff_day'] = (df['fault_time'] - df['dt']).dt.days
        df['label'] = 0
        df.loc[(df['diff_day']>=0)&(df['diff_day']<=30),'label'] = 1
        return df

    #2017
    for i in range(11,12):
        gc.collect()
        train_2017 = pd.read_csv('./data/disk_sample_smart_log_2017'+str(i)+'.csv')
        train_2017 = get_label(train_2017)
        joblib.dump(train_2017, './data/train_2017_'+str(i)+'.jl.z')

    #2018
    for i in range(1,8):
        gc.collect()
        train_2018 = pd.read_csv('./data/disk_sample_smart_log_20180'+str(i)+'.csv')
        train_2018 = get_label(train_2018)
        joblib.dump(train_2018, './data/train_2018_'+str(i)+'.jl.z')

###提取出每个硬盘最早出现的时间日期，内存不够的话，只能一个一个读取
train_2018_7 = joblib.load('./data/train_2018_7.jl.z')
train_2018_6 = joblib.load('./data/train_2018_6.jl.z')
train_2018_5 = joblib.load('./data/train_2018_5.jl.z')
train_2018_4 = joblib.load('./data/train_2018_4.jl.z')
train_2018_3 = joblib.load('./data/train_2018_3.jl.z')
train_2018_2 = joblib.load('./data/train_2018_2.jl.z')
train_2018_1 = joblib.load('./data/train_2018_1.jl.z')

train_2017_7 = joblib.load('./data/train_2017_7.jl.z')
train_2017_8 = joblib.load('./data/train_2017_8.jl.z')
train_2017_9 = joblib.load('./data/train_2017_9.jl.z')
train_2017_10 = joblib.load('./data/train_2017_10.jl.z')
train_2017_11 = joblib.load('./data/train_2017_11.jl.z')
train_2017_12 = joblib.load('./data/train_2017_12.jl.z')

serial_2017_7 = train_2017_7[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2017_8 = train_2017_8[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2017_9 = train_2017_9[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2017_10 = train_2017_10[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2017_11 = train_2017_11[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2017_12 = train_2017_12[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')

serial_2018_1 = train_2018_1[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2018_2 = train_2018_2[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2018_3 = train_2018_3[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2018_4 = train_2018_4[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2018_5 = train_2018_5[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2018_6 = train_2018_6[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2018_7 = train_2018_7[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')
serial_2018_8 = test[['serial_number','dt']].sort_values('dt').drop_duplicates('serial_number')

serial = pd.concat((serial_2017_7,serial_2017_8,serial_2017_9,serial_2017_10,serial_2017_11,serial_2017_12),axis = 0)
serial = pd.concat((serial,serial_2018_1,serial_2018_2,serial_2018_3,serial_2018_4,serial_2018_5,serial_2018_6,serial_2018_7,serial_2018_8),axis = 0)
serial = serial.sort_values('dt').drop_duplicates('serial_number').reset_index(drop=True)

serial.columns = ['serial_number','dt_first']
serial.dt_first = pd.to_datetime(serial.dt_first)
serial.to_csv("data/serial.csv",index=False)




