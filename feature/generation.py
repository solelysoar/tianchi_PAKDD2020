#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import gc

from datetime import timedelta


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


def day_tag(x, holidays):
    """
    返回该天的特征，1表示节日非周末，2表示周末非节日，3表示节日且假日，4表示普通工作日
    x: datetime
    """
    weekly_label = x.isoweekday()  # 1代表周一，7代表周日
    if x in holidays and weekly_label<6:
        return 1
    elif weekly_label >= 6 and x not in holidays:
        return 2
    elif x in holidays and weekly_label >= 6:
        return 3
    else:
        return 4


def build_feature(df, day_ahead, first_day, tag, ori_fea_list):
    """
    针对df对部分特征进行transform，并生成新的衍生特征
    :param df: 原始数据
    :param day_ahead: 生成slope特征时取day_ahead前的特征作差
    :param first_day: 每个硬盘对应的启用时间的DataFrame
    :param tag: tag文件的DataFrame
    :param ori_fea_list: 挑选的原始特征列表
    :return: 加入了新特征的数据
    """
    # 特征定义
    log_features = [i for i in ori_fea_list if "raw" in i]
    holidays = pd.to_datetime(["2017-7-1", "2017-9-1", "2017-10-01", "2017-10-08", "2017-11-1", "2017-12-30",
                               "2018-01-01", "2018-02-15", "2018-02-21", "2018-3-2", "2018-04-05", "2018-04-07",
                               "2018-04-29", "2018-05-01", "2018-06-16", "2018-06-18", "2018-7-1", "2018-8-1",
                               "2018-09-1", "2018-09-22", "2018-09-24", "2018-10-01", "2018-10-07"]).to_list()

    # 特征提取
    df = df.merge(first_day, how='left', on=["manufacturer", "model", "serial_number"])
    print("特征做log变换")
    for column in log_features:
        df[column] = df[column].apply(lambda x: np.log1p(x))
    # 增加特征：硬盘的使用时常
    df['days'] = (df['dt'] - df['dt_first']).dt.days
    df['days'] = df['days'].astype("category")
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
    df_shift = df.copy()
    df_shift["dt"] = df_shift["dt"].apply(lambda x: x + timedelta(days=day_ahead))
    df_shift = df_shift[[i for i in df.columns if "smart" in i] + ["manufacturer", "model", "serial_number", "dt"]]
    df_shift.columns = [i + '_shift' if "smart" in i else i for i in df_shift.columns]  # 重命名
    df = df.merge(df_shift, on=["manufacturer", "model", "serial_number", "dt"], how="left")
    del df_shift
    gc.collect()
    for feature in [i for i in ori_fea_list if i not in ['dt', 'manufacturer', 'model', 'serial_number']]:
        df[feature + "_slope"] = df[feature] - df[feature + "_shift"]

    # serial_number 标签化
    df['serial_number'] = df['serial_number'].apply(lambda x: int(x.split('_')[1]))
    df['serial_number'] = df['serial_number'].astype("category")

    return df
