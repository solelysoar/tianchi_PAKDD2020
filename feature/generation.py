#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import gc
import swifter

from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")


def distance_to_next_holiday(x, holidays):
    """
    返回日期x距离下一次最近的节假日的天数
    x: datetime
    """
    # 返回最小的正数
    distance = np.array([(day - x).days for day in holidays if day > x])
    return distance.min()


def distance_to_last_holiday(x, holidays):
    """
    返回日期x距离上一次最近的节假日的天数
    x: datetime
    """
    # 返回最小的正数
    distance = np.array([(x - day).days for day in holidays if day < x])
    return distance.min()


def build_feature(df, day_ahead, ori_fea_list, slop_features):
    """
    针对df对部分特征进行transform，并生成新的衍生特征
    :param df: 原始数据
    :param day_ahead: list, 生成slope特征时取day_ahead前的特征作差, 如果存在多个数字则求平均
    :param slop_features: 每个硬盘对应的启用时间的DataFrame
    :param divide_features: tag文件的DataFrame
    :param ori_fea_list: 挑选的原始特征列表
    :return: 加入了新特征的数据
    """
    # 读取需要的文件
    tag = pd.read_csv('./user_data/tmp_data/disk_sample_fault_concat_tag.csv')  # <----改路径！
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])
    tag['tag'] = tag['tag'].astype(str)
    tag = tag.groupby(['serial_number', 'fault_time', 'model'])['tag'].apply(lambda x: '|'.join(x)).reset_index()
    tag.columns = ['serial_number', 'fault_time_1', 'model', 'tag']

    first_day = pd.read_csv('./user_data/tmp_data/first_use_day_updated.csv')  # <----改路径！
    first_day.dt_first = pd.to_datetime(first_day.dt_first)

    fault_disk_dt_last = pd.read_csv('./user_data/tmp_data/fault_disk_dt_last_updated.csv')
    fault_disk_dt_last.dt_last = pd.to_datetime(fault_disk_dt_last.dt_last)

    # 特征定义
    log_features = [i for i in ori_fea_list if "raw" in i]
    holidays = pd.to_datetime(["2017-7-1", "2017-9-1", "2017-10-01", "2017-10-08", "2017-11-1", "2017-12-30",
                               "2018-01-01", "2018-02-15", "2018-02-21", "2018-3-2", "2018-04-05", "2018-04-07",
                               "2018-04-29", "2018-05-01", "2018-06-16", "2018-06-18", "2018-7-1", "2018-8-1",
                               "2018-09-1", "2018-09-22", "2018-09-24", "2018-10-01", "2018-10-07"]).to_list()

    # 特征提取
    df = df.merge(first_day, how='left', on=["manufacturer", "model", "serial_number"])
    # 数零值
    # zero_count_features = ["smart_5raw", "smart_187raw", "smart_188raw",
    # "smart_189raw", "smart_197raw", "smart_198raw"]
    # for feature in zero_count_features:
    #     df[feature + "_above_zero"] = df[feature].apply(lambda x: 1 if x > 0 else 0)
    # df["above_zero_count"] = df[[i + "_above_zero" for i in zero_count_features]].sum(axis=1)
    # # 增加除法特征
    # for feature in divide_features:
    #     df[feature + "_divide"] = df[feature + "raw"] / (df[feature + "_normalized"] + 0.1)
    print("特征做log变换")
    for column in log_features:
        df[column] = df[column].apply(lambda x: np.log1p(x))
    # 增加特征：硬盘的使用时常
    df['days'] = (df['dt'] - df['dt_first']).dt.days
    df = df.merge(tag, how='left', on=['serial_number', 'model'])
    # 增加特征：dt对应的月份、周几、是否节假日
    print("加入日期特征")
    df['month'] = df['dt'].apply(lambda x: np.cos(x.month * np.pi / 6))
    df["days_to_next_holiday"] = df["dt"].swifter.progress_bar(enable=True).apply(lambda x:
                                                                                  distance_to_next_holiday(x, holidays))
    df["days_to_last_holiday"] = df["dt"].swifter.progress_bar(enable=True).apply(lambda x:
                                                                                  distance_to_last_holiday(x, holidays))
    df["day_position"] = df["days_to_last_holiday"] / (df["days_to_last_holiday"] + df["days_to_next_holiday"])
    # 增加特征：相比于3天前数据变化的斜率
    print("加入slope特征")
    for feature in slop_features:
        df[feature + "_slope"] = 0
    weights = [1 / len(day_ahead) for k in range(len(day_ahead))]  # 现在是平均的weights
    for index, n_ahead in enumerate(day_ahead):
        print("加入{}天前的slope".format(n_ahead))
        df_shift = df.copy()
        df_shift["dt"] = df_shift["dt"].apply(lambda x: x + timedelta(days=n_ahead))
        df_shift = df_shift[slop_features + ["manufacturer", "model", "serial_number", "dt"]]
        df_shift.columns = [i + '_shift_' + str(n_ahead) if "smart" in i else i for i in df_shift.columns]  # 重命名
        df = df.merge(df_shift, on=["manufacturer", "model", "serial_number", "dt"], how="left")
        del df_shift
        gc.collect()
        for feature in slop_features:
            df[feature + "_slope"] += (df[feature] - df[feature + "_shift_" + str(n_ahead)]) * weights[index]
        df = df.drop([i for i in df.columns if "shift" in i], axis=1)

    # serial_number 标签化
    df['serial_number'] = df['serial_number'].apply(lambda x: int(x.split('_')[1]))
    df['serial_number'] = df['serial_number'].astype("category")
    df['model'] = df['model'].astype("category")

    return df
