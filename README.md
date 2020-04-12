# PAKDD2020
**队伍名称：东方低碳-旷视科技**  
**成员：旺仔牛奶的糖，闪电云游，wentixiaoge**

---

#### 所需安装包和版本号：
python = 3.6.8  
pandas = 0.25.3  
numpy = 1.18.1  
tqdm = 4.43.0  
swifter = 0.301  
joblib = 0.14.1  
lightgbm  = 2.3.1    

## 1. 程序结构
|--data   原始数据文件  

    |--round1_train  
        |--*other smart log files
    |--disk_sample_smart_log_round2
        |--*test files for round2
|--user_data 用户数据文件夹

     |--tmp_data
         |--disk_sample_fault_concat_tag.csv
         |--fault_dist_dt_last_updated.csv
         |--first_use_day_updated.csv
|--feature  特征工程文件夹

    |--extraction_data.py
    |--generation.py
|--model  模型训练文件夹

    |--model_saved
        |--*pkl file for multiple models (ensembling)
|--prediction_result

    |--predictions.csv
|--code  代码文件夹

    |--main.py
    |--requirements.txt    
|--README.md

## 2. 运行指南
step 1: 将官方提供的训练数据和测试数据分别放入 ./data 下的各自文件夹内  
step 2: 在代码主目录下执行 python -m code.main 即可执行。若使用IDE，需要将工作目录改为主目录  

详细解释：  
本程序以 ./code/main.py 为主入口程序，只需运行main.py会自动调用其他文件夹下的相关程序，例如：  
from feature.generation import build_feature  
from feature.extraction_data import prepare_tmp_data  
程序会自动检测是否存在生成的中间数据，对于不存在的中间数据自动进行中间数据的生成，可实现多次运行的功能；

## 3. 特征生成
1. 算法中运用到的数据特征主要通过extraction_data.py和generation.py两个文件生成。  
    1. extraciton_data.py主要是抽取训练集文件中的重要原始特征，舍去全为空值或特殊值只有一个的特征，同时去掉标记过程中产生的噪音点。  
    2. generation.py是在训练集原始数据特征的基础上，提取一些和故障强相关的数据特征：
硬盘使用的天数、月份、当前时间距离上一次最近的节假日的天数、当前时间距离下一次最近节假日的天数、
节假日打标、slope特征（原始特征与n天前的差）等；

2. 在提取数据特征的同时，我们也对数据集做了数据预处理：对raw的特征数据做log变换、对月份标签做cos变换；

3. 我们尝试了很多其他特征，例如对多个smart特征统计超过0值的个数，对smart特征的raw和normalized取相除特征（借鉴华为硬盘测试比赛
冠军方案），但发现新特征与原始特征可能有相互排斥的现象，会恶化线上成绩。由于复赛递交次数有限，无法做完备的数据实验来证明。

3. 不同于初赛使用的单个lightgbm模型，为了增强模型的稳定性与泛化性能，我们使用了多模型投票的方式。通过选取两组训练数据（
分别为2018年5、6月数据，和2018年6、7月数据）搭配不同的lgb参数组合，训练了14个子模型。由于不同的模型在预测概率时概率值本身
有较大偏差，对概率结果进行加权平均的方式显然不可取，于是对单个模型也进行阈值筛选，只有超过一半的模型预测出故障时才将该盘纳入
结果。用此方法不仅增强了稳定性，同时使模型对单个阈值的依赖性降低，不需要阈值过分精确。

4. 在训练模型时我们也尝试了不断加入更多的数据，然而相比于2018年，2017年的数据过少，加入后会损失模型效果，即使是加入2018年4月
的数据也会使得模型效果变差，我们推测硬盘的总数量级在不断增多，分布和特性也在不断变化，因此只有取得最近的月份训练才能产生
有效的结果。

## 4. 内存
本次比赛中数据集较大，在进行计算和训练模型时对读入数据进行了优化。首先在extraction_data.py中读取csv时
采用了分chunk读取的方式，处理后的数据使用joblib进行存储，便于之后的高效读取。整个程序从处理原始数据到
完成结果预测大约花费1小时~1.5小时。由于在生成特征时需要计算一些中间值，建议使用大一些的内存（40G以上）

## 5. 致谢
感谢阿里天池提供了数据集，比赛的过程很短暂，有很多想法都是浅尝辄止，例如对数据样本进行downsampling降低不平衡率，
使用Tomeklinks去除交界处健康样本以便于分类等，受限于提交次数无法得出最终结论，希望我们的工作能带来哪怕一点点的
启发，能帮助主办方离更好的解决方案更近。
