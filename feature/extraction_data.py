import numpy as np
import pandas as pd
import gc
import os
import joblib


def prepare_tmp_data(tmp_file_needed, ori_fea_list):
    tag = pd.read_csv('../data/round1_train/disk_sample_fault_tag.csv')
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])

    # tag表里面有的硬盘同一天发生几种故障， 删掉多余的记录
    tag = tag.drop_duplicates(["manufacturer", "model", "serial_number"])

    # 判断哪些数据需要处理
    exist_tmp_file = os.listdir("../user_data/tmp_data")
    process_file = [i for i in tmp_file_needed if i not in exist_tmp_file]
    for file in process_file:
        # 分chunk_size读取，避免内存需量不足
        chunk_size = 10 ** 6
        filename = os.path.join('../data/round1_train', get_ori_data_file_name(file))
        ori_data = pd.DataFrame([])
        for chunk in pd.read_csv(filename, chunksize=chunk_size):
            ori_data = pd.concat([ori_data, chunk], axis=0, sort=False)
        data = ori_data[ori_fea_list]
        del ori_data
        gc.collect()
        data = get_label(data, tag)
        joblib.dump(data, os.path.join('../user_data/tmp_data', file))
        print("finished_process: {}".format(file))
        del data
        gc.collect()


def get_label(df, tag):
    df['dt'] = pd.to_datetime(df['dt'], format="%Y%m%d")
    df = df.merge(tag, how='left', on=['manufacturer', 'model', 'serial_number'])
    df['diff_day'] = (df['fault_time'] - df['dt']).dt.days
    df['label'] = 0
    df.loc[(df['diff_day'] >= 0) & (df['diff_day'] <= 30), 'label'] = 1
    df = df.drop(["diff_day", "tag"], axis=1)  # 去掉不必要的列
    return df


def get_ori_data_file_name(x):
    """
    给定一个需要的文件名，如train_2017_8,返回原始数据的文件名disk_sample_smart_log_201708.csv
    :param x:
    :return:
    """
    year = x.split("_")[1]
    month = x.split("_")[-1].split(".")[0]
    if int(month) < 10:
        month = "0" + month
    return "disk_sample_smart_log_{}.csv".format(year+month)




