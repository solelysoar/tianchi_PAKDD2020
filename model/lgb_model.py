#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import joblib
import gc
from feature.generation import build_feature


def train_model(train_data, val_data, clf, n_ahead, ori_fea_list):
    '''
    用于训练模型时训练集和验证集不一样的情况
    '''
    # 准备数据
    train_data = build_feature(train_data, ori_fea_list=ori_fea_list)
    train_y = train_data["label"].values
    train_x = train_data.drop(["label"], axis=1)
    val_data = build_feature(val_data, ori_fea_list=ori_fea_list)
    val_y = val_data["label"].values
    val_x = val_data.drop(["label"], axis=1)

    clf = train(clf, train_x, train_y, val_x, val_y, n_early_stop=150, n_verbose=100)

    # 保存模型
    joblib.dump(clf, './model_saved/lgb_1.pkl')

    del train_x, train_y, val_x, val_y
    gc.collect()

    return clf


def train(clf, train_x, train_y, val_x, val_y, n_early_stop, n_verbose):
    print('************** training **************')
    print(train_x.shape, val_x.shape)
    clf.fit(
        train_x, train_y,
        eval_set=[(val_x, val_y)],
        eval_metric='auc',
        early_stopping_rounds=n_early_stop,
        verbose=n_verbose
    )
    return clf
