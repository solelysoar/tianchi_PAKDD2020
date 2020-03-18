# PAKDD2020

#### 所需安装包和版本号：
python = 3.6.8  
pandas = 0.25.3  
numpy = 1.18.1  
tqdm = 4.43.0  
swifter = 0.301  
joblib = 0.14.1  
lightgbm  = 2.3.1  
dask == 2.11.0  

## 1. 程序结构
|--data   原始数据文件  

    |--round1_train  
        |--disk_sample_fault_tag.csv
        |--*other smart log files
    |--round1_testA
        |--disk_sample_smart_log_test_a.csv
    |--round1_testB
        |--disk_sample_smart_log_test_b.csv
|--user_data 用户数据文件夹

     |--tmp_data
         |--fault_dist_dt_last.csv
         |--first_use_day.csv
|--feature  特征工程文件夹

    |--extraction_data.py
    |--generation.py
|--model  模型训练文件夹

    |--lgb_model.py
|--prediction_result

    |--predictions.csv
|--code  代码文件夹

    |--main.py
    |--requirements.txt    
|--README.md

## 2. 运行指南
step 1: 将官方提供的训练数据和测试数据分别放入 ./data 下的各自文件夹内
step 2: 在代码主目录下执行 python -m code.main 即可启动
本程序以 ./code/main.py 为主入口程序，只需运行main.py会自动调用其他文件夹下的相关程序，例如：
from feature.generation import build_feature
from feature.extraction_data import prepare_tmp_data
from model.lgb_model import train
程序会自动检测生成的中间数据，可实现多次运行的功能；

## 3. 特征生成
1. 算法中运用到的数据特征主要通过extraction_data.py和generation.py两个文件生成。extraciton_data.py主要是抽取训练集文件中的重要特征，generation.py是在训练集原始数据特征的基础上，提取一些和故障强相关的数据特征：距离上一次最近的节假日的天数、距离下一次最近节假日的天数、节假日打标、slope特征等；

2. 在提取数据特征的同时，我们也对数据集做了数据预处理：去除基本为空的特征、去除偏大和偏小的噪音数据、对较大的数据做log变换、对月份标签做cos变换；

3. 在训练模型时我们总共有两套模型：增量模型和普通模型，这两套模型都是应用lightgbm进行建模。增量模型训练了历史上每两个月的数据，并用隔一个月的数据作为验证集；普通模型训练了5、6月份的数据，并用7月份数据作为验证集。其中普通模型对于准确度的提升帮助比较大。
