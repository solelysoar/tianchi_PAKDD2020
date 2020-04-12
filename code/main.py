#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import joblib
import gc
import os
from datetime import datetime, timedelta
from lightgbm.sklearn import LGBMClassifier

from feature.generation import build_feature
from feature.extraction_data import prepare_tmp_data
import warnings
warnings.filterwarnings("ignore")


def read_data(month_list, only_positive_month=[]):
    data = pd.DataFrame([])
    for month_read in month_list:
        next_df = joblib.load('./user_data/tmp_data/train_' + month_read + '.jl.z')  # <----改路径！
        if month_read in only_positive_month:
            next_df = next_df[next_df["label"] == 1]
        data = pd.concat([data, next_df], axis=0, sort=False)
        del next_df
    gc.collect()
    return data


def update_dt_first(df_test, new_disk_list):
    """
    根据新加入的数据不断更新dt_first文件，并统计加入的新盘数量
    :param df_test:
    :param new_disk_list:
    :return:
    """
    first_day = pd.read_csv('./user_data/tmp_data/first_use_day_updated.csv')  # <----改路径！
    first_day.dt_first = pd.to_datetime(first_day.dt_first)
    first_day["unique_disk_id"] = first_day.swifter.progress_bar(enable=False).apply(
        lambda x: "{}_{}_{}".format(x["manufacturer"], x["model"], x["serial_number"]), axis=1
    )
    df_test["unique_disk_id"] = df_test.swifter.progress_bar(enable=False).apply(
        lambda x: "{}_{}_{}".format(x["manufacturer"], x["model"], x["serial_number"]), axis=1
    )
    newly_added_disks = set(df_test["unique_disk_id"].unique()).difference(set(first_day["unique_disk_id"].unique()))
    print("find new disks: {}".format(len(newly_added_disks)))
    print(newly_added_disks)
    new_disk_list += list(newly_added_disks)
    if len(newly_added_disks) > 0:
        _new_first_day = []
        for disk in newly_added_disks:
            _manufacturer = disk.split("_")[0]
            _model = disk.split("_")[1]
            _serial_number = "disk_" + disk.split("_")[-1]
            _dt_first = df_test.loc[df_test["unique_disk_id"] == disk, "dt"].min()
            _new_first_day.append([_manufacturer, _model, _serial_number, _dt_first, disk])
        _new_first_day = pd.DataFrame(_new_first_day, columns=["manufacturer", "model", "serial_number", "dt_first",
                                                               "unique_disk_id"])
        first_day = first_day.append(_new_first_day, sort=False)
        first_day[["manufacturer", "model", "serial_number", "dt_first"]].to_csv(
            './user_data/tmp_data/first_use_day_updated.csv', index=False)  # <----改路径！
        print("finish update dt_first file")

    return new_disk_list


def predict(df_test, current_date):
    df_test = df_test.sort_values(['model', 'serial_number', 'dt'])
    df_test = df_test.drop_duplicates().reset_index(drop=True)
    df_test = build_feature(df_test, n_ahead, ori_fea_list, slope_features)
    df_submit = df_test[['manufacturer', 'model', 'serial_number', 'dt', 'days', 'unique_disk_id']]
    df_test_x = df_test[total_features]
    print("category features: {}".format(df_test_x.columns[df_test_x.dtypes == "category"]))

    n_model = 1
    threshold = [0.0077 for i in range(n_model)]
    print(threshold)
    df_submit["voting"] = 0
    for n in range(n_model):
        model = joblib.load("./model/model_saved/lgb_voting_{}.pkl".format(n))
        df_submit["p_{}".format(n)] = model.predict_proba(df_test_x)[:, 1]
        df_submit["detected_{}".format(n)] = df_submit["p_{}".format(n)].apply(lambda x: 1 if x >= threshold[n] else 0)
        df_submit["voting"] += df_submit["detected_{}".format(n)]

    return df_submit[(df_submit["dt"] == current_date) & (df_submit["voting"] > n_model / 2)]


if __name__ == '__main__':
    # 筛选后的原始特征顺序
    ori_fea_list = ['serial_number',
                    'model',
                    'manufacturer',
                    'dt',
                    'smart_10_normalized',
                    'smart_184raw',
                    'smart_187_normalized',
                    'smart_187raw',
                    'smart_188_normalized',
                    'smart_188raw',
                    'smart_189_normalized',
                    'smart_189raw',
                    'smart_192raw',
                    'smart_193raw',
                    'smart_194raw',
                    'smart_195_normalized',
                    'smart_198raw',
                    'smart_199raw',
                    'smart_1_normalized',
                    'smart_1raw',
                    'smart_240raw',
                    'smart_241raw',
                    'smart_242raw',
                    'smart_3_normalized',
                    'smart_5raw',
                    'smart_7_normalized',
                    'smart_7raw',
                    'smart_9raw']
    slope_features = [i for i in ori_fea_list if "smart" in i]
    total_features = [i for i in ori_fea_list if i not in ['dt', 'manufacturer']] + \
                     [i+'_slope' for i in slope_features] + \
                     ['days', 'month', 'days_to_next_holiday', 'days_to_last_holiday']

    # 判断是否需要进行临时数据的准备
    clean_file_list = ["train_2018_5.jl.z", "train_2018_6.jl.z", "train_2018_7.jl.z"]
    exist_files = os.listdir("./user_data/tmp_data")
    to_clean = [1 if file not in exist_files else 0 for file in clean_file_list]
    if sum(to_clean) != 0:
        prepare_tmp_data(clean_file_list, ori_fea_list)

    # 参数定义
    n_ahead = [1, 2, 3]  # 对这几天前的slope特征取平均
    max_ahead = max(n_ahead) + 1
    start_time = datetime.now()  # 计时用的

    # print("start reading data for training...")
    # data_list = []
    # train_data = read_data(["2018_5", "2018_6"], only_positive_month=[])
    # train_data = build_feature(train_data, n_ahead, ori_fea_list, slope_features)
    # data_list.append(train_data)
    # train_data = read_data(["2018_6", "2018_7"], only_positive_month=[])
    # train_data = build_feature(train_data, n_ahead, ori_fea_list, slope_features)
    # data_list.append(train_data)
    # del train_data
    # gc.collect()
    #
    # params = [
    #     {
    #         "learning_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.01, 0.1],
    #         "n_estimators": [100, 150, 200, 100, 100, 100, 100],
    #         "num_leaves": [127, 127, 127, 65, 255, 127, 127]},
    #     {
    #         "learning_rate": [0.001, 0.001, 0.001, 0.001, 0.001, 0.01, 0.1],
    #         "n_estimators": [100, 150, 200, 100, 100, 100, 100],
    #         "num_leaves": [127, 127, 127, 65, 255, 127, 127]}
    # ]  # 第一组变量是给第一组数据的，第二组变量组合是给第二组数据的
    # model_index = 0
    # for i in range(len(data_list)):
    #     print("this time use No.{} data".format(i))
    #     train_data = data_list[i]
    #     train_y = train_data["label"].values
    #     train_x = train_data[total_features]
    #     del train_data
    #     gc.collect()
    #
    #     for j in range(len(params[i]["learning_rate"])):
    #         i_params = params[i]
    #         print("this time training: learning_rate = {}, n_estimators = {}, num_leaves = {}".format(
    #             i_params["learning_rate"][j], i_params["n_estimators"][j], i_params["num_leaves"][j]))
    #         print("start training model_{}".format(model_index))
    #         clf = LGBMClassifier(
    #             learning_rate=i_params["learning_rate"][j],
    #             n_estimators=i_params["n_estimators"][j],
    #             num_leaves=i_params["num_leaves"][j],
    #             subsample=0.8,
    #             colsample_bytree=0.8,
    #             random_state=2019,
    #             is_unbalenced='True',
    #             metric=None)
    #
    #         print('************** training **************')
    #         print(train_x.shape)
    #         clf.fit(
    #             train_x, train_y,
    #             eval_set=[(train_x, train_y)],
    #             eval_metric='auc',
    #             early_stopping_rounds=10,
    #             verbose=10
    #         )
    #         # 保存模型
    #         joblib.dump(clf, './model/model_saved/lgb_voting_{}.pkl'.format(model_index))
    #         model_index += 1

    # 预测部份
    test_data_dir = "./data/disk_sample_smart_log_round2"
    test_file_list = os.listdir(test_data_dir)

    submit = pd.DataFrame([])
    new_disks = []
    for day in pd.date_range("2018-08-20", "2018-09-30"):
        print("start predicting for {}".format(day.strftime("%Y-%m-%d")))
        test = pd.DataFrame([])
        retrieve_day = [(day - timedelta(days=max_ahead - i)).strftime("%Y%m%d") for i in range(max_ahead + 1)]
        for read_day in retrieve_day:
            file_path = os.path.join(test_data_dir, "disk_sample_smart_log_{}_round2.csv".format(read_day))
            if os.path.exists(file_path):
                next_read = pd.read_csv(file_path)
                test = pd.concat([test, next_read], axis=0, sort=False)
        if test.empty:
            continue
        test["dt"] = pd.to_datetime(test["dt"], format="%Y%m%d")
        # 更新dt_first文件
        new_disks = update_dt_first(test, new_disks)
        print("accumulated new disks: {}".format(len(new_disks)))
        next_predict = predict(test, day)
        print(next_predict.shape)
        submit = pd.concat([submit, next_predict], axis=0, sort=False)
    print("total submit disk no: {}".format(len(submit.drop_duplicates(["manufacturer", "model", "serial_number"]))))

    # 保存结果
    submit[['manufacturer', 'model', 'serial_number', 'dt']].to_csv(
        "./prediction_result/predictions.csv", index=False, header=None)
    print("prediction done, please head to prediction_result folder to check!")
    print("total time used: {}".format(datetime.now() - start_time))
