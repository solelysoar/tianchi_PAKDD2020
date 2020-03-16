#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import joblib
import gc
from datetime import timedelta
from lightgbm.sklearn import LGBMClassifier

from feature.generation import build_feature
from model.lgb_model import train


def read_data(month_list):
    data = joblib.load('../user_data/tmp_data/train_' + month_list[0] + '.jl.z')  # <----改路径！
    for i_read in range(1, len(month_list)):
        next_df = joblib.load('../user_data/tmp_data/train_' + month_list[i_read] + '.jl.z')  # <----改路径！
        data = pd.concat([data, next_df], axis=0, sort=False)
        del next_df
    gc.collect()
    return data


def read_submit_test_data(day_ahead):
    df_2018_7 = joblib.load("../user_data/tmp_data/train_2018_7.jl.z")  # <----改路径！
    df_2018_7 = df_2018_7[df_2018_7["dt"] >= df_2018_7["dt"].max() - timedelta(days=day_ahead)]

    df_test = pd.read_csv('../data/testA/disk_sample_smart_log_test_a.csv')  # <----改路径！
    df_test['dt'] = pd.to_datetime(df_test['dt'], format="%Y%m%d")
    disk_mark = df_test.drop_duplicates(["manufacturer", "model", "serial_number"])
    disk_mark["in_test"] = 1
    disk_mark = disk_mark.loc[:, ["manufacturer", "model", "serial_number", "in_test"]]

    df_test = df_2018_7.append(df_test, sort=False)
    del df_2018_7
    gc.collect()

    # 删除掉不应该在里面的部分
    df_test = df_test.merge(disk_mark, on=["manufacturer", "model", "serial_number"], how="left")
    df_test = df_test[df_test["in_test"] == 1]
    df_test = df_test.drop(["in_test"], axis=1)
    df_test = df_test[df_test['dt'] >= "2018-07-31"]
    return df_test


if __name__ == '__main__':
    # 筛选后的原始特征顺序
    ori_fea_list = ['smart_7raw',
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

    # 参数定义
    n_ahead = 3

    train_data = read_data(["2018_5", "2018_6"])
    train_data = build_feature(train_data, n_ahead, ori_fea_list)
    train_y = train_data["label"].values
    train_x = train_data.drop(["label"], axis=1)

    model = LGBMClassifier(
        learning_rate=0.001,
        n_estimators=100,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2019,
        is_unbalenced='True',
        metric=None)

    model = train(model, train_x, train_y, val_x=train_x, val_y=train_y, n_early_stop=10, n_verbose=10)

    test = read_submit_test_data(n_ahead)
    test = test.sort_values(['serial_number', 'dt'])
    test = test.drop_duplicates().reset_index(drop=True)

    sub = test[['manufacturer', 'model', 'serial_number', 'dt']]
    test_x = build_feature(test, n_ahead, ori_fea_list)

    result = model.predict_proba(test_x)[:, 1]
    sub['p'] = result
    # 提交
    p_threshold = 0.0034
    submit = pd.DataFrame([])  # 结果初始化
    for day in pd.date_range("2018-08-01", "2018-08-31"):
        result_today = sub[sub["dt"] == day]
        submit = submit.append(result_today[result_today["p"] >= p_threshold], sort=False)
    submit = submit.drop_duplicates(['serial_number', 'model'])
