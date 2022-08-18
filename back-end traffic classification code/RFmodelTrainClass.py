'''
主函数为：RFModelClass(labeledDataDf,pklPath,featuresMy)
其他函数要实现已标记数据的RF model的训练，直接调用函数RFModelClass()
'''

import numpy as np
import pandas as pd
from collections import Counter
from scipy import linalg
from sklearn.cluster import AgglomerativeClustering
import time
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import ensemble
from BigSmallLabel import BigSmalllabel
import joblib


#******************************************************************RF model 的训练部分****************************************************************************
#------------------------------------------------数据处理函数-----------------------------------------------------------------------------------------------------
def AvgOfFNR(FNRList):# FNR=FN/(TP+FN)  会出现单个的nan值 无法求均值
    sum = 0;
    count = 0;
    for i in range(0,len(FNRList)):
        PerFNR=FNRList[i]
        if(np.isnan(PerFNR)!= True):#是nan为True
            sum=sum+PerFNR
            count=count+1
    return (sum*1.0)/count
#------------------------------------------------模 型 函 数------------------------------------------------------------------------------------------------------
def try_method(model,data,dataframe,labeledPath,startTime):
    """
    :param model: 算法模型
    :param data: 数据集
    :param dataframe:数据集对应的原dataframe
    :param labeledPath: 训练好的已标记数据文件路径 可以识别是1TD/2TS
    :param startTime: 主函数中启动本次模型构建过程的开始时间
    :return: 分类结果和plt可视化图形
    """

    # 获取dataFrame的行数 dataframe.shape[0]；列数 dataframe.shape[1].
    x = data[:, 0:(data.shape[1] - 1)] #取出特征所在列 特征从 C1到C14.
    y = data[:, (data.shape[1] - 1)]   #取出标签所在  最后的一列
    y = y.astype('int') #标签数据集的类型出错，sklearn识别前需要转换.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  # 30%测试集

    model.fit(x_train, y_train) #模型训练：使用传入的已有模型进行数据的拟合

    score = model.score(x_test, y_test) #准确度=对角线上的数值加总 /总数;     显示在图上的准确度（正确率）=所有预测正确的样本/总的样本 （TP+TN）/总
    result = model.predict(x_test)

    # 分别计算各个类别的精确率与召回率，只记录大类的指标.
    print('2 开始计算大类的精确度与召回率，并按照“全部训练集” 中的类别占比 降序输出-------')

    # 返回bigLabel:整个训练集中大类的label, 返回顺序是按照占比从大到小.
    # 返回bigLabelPro:整个训练集中大类的占比, 返回顺序同bigLabel.
    bigLabel,bigLabelPro= BigSmalllabel(dataframe)
    label_result=[] #记录每个label相应性能指标 precison recall=[[label1,20%,0.99,0.98 ],[label2,5%,0.98,0.97],..]
    column = ['Biglabel', 'Proportion', 'FPR', 'FNR']
    label_result.append(column)
    for i in range(0,len(bigLabel),1):
        TP1 = 0  # 实际上是label1，预测也是label1 （TP）
        TN1 = 0  # 实际上不是label1，预测也不是label1(TN)
        FN1 = 0  # 实际上是label1，预测不是label1（FN）
        FP1 = 0  # 实际上不是label1，预测却是label1
        row=[]
        targetlabel1=bigLabel[i]
        for j in range(0,len(y_test),1):
            if(y_test[j]==targetlabel1 and result[j]==targetlabel1):#当前实际label为测试集中的目标label.
                TP1=TP1+1
                # print('TP1: ',targetlabel1,y_test[j],result[j])
            elif(y_test[j]==targetlabel1 and result[j]!=targetlabel1):#实际上是目标label，但是预测不是目标label
                FN1=FN1+1
                # print('---------------------FN1: ',targetlabel1, y_test[j], result[j])
            if(y_test[j]!=targetlabel1 and result[j]!=targetlabel1):#当前实际label不是目标label，预测label也不是目标label
                TN1=TN1+1
            elif(y_test[j]!=targetlabel1 and result[j]==targetlabel1):#当前实际label不是目标label，但是预测label是目标label
                FP1=FP1+1
                # print('---------------------FP1: ', targetlabel1, y_test[j], result[j])

        if(TN1 + FP1 ==0):
            FPR1 =0;
        else:
            FPR1 = FP1 / (TN1 + FP1)  # 大类的假阳率
        if(TP1 + FN1==0):
            FNR1 =0;
        else:
            FNR1 = FN1 / (TP1 + FN1)  # 大类的假阴率

        row.append(targetlabel1) #0 为大类的label
        row.append(bigLabelPro[i]) #1 为此label的占比
        row.append(FPR1) #2 FPR1
        row.append(FNR1) #3 FNR1

        label_result.append(row)


    #-----------------------------------------------------计算Micro F1  和  Macro F1-------------------------------------------------------------------------------------------
    # Micro F1: 将n分类的评价拆成n个二分类的评价，将n个二分类评价的TP、FP、RN对应相加，计算平均准确率和召回率，由这2个准确率和召回率计算的F1 score即为Micro F1。
    Mi_cro_f1 = f1_score(y_test, result, average="micro")

    #-------------------------------------------------------将所需指标一次性写入文件---------------------------------------------------------------------------------------------
    Biglabel_FPR = []
    Biglabel_FNR = []
    for i in range(1, len(label_result), 1):  # 从第二个元素计数  第一个元素为列名.
        row = label_result[i]  # [label2,占比,FPR,FNR]
        if (row[1] >= 0.03):  # 只将占比大于0.03=0.3%的类抽出记为结果中的大类.  0.03大约5类   0.003大约10~15类.
            Biglabel_FPR.append(row[2])
            Biglabel_FNR.append(row[3])

    performanceResult=[]#记录评价指标 ：文件来源记住, 全部类的平均精确率，全部类的平均召回率，大类的平均精确率，大类的平均召回率,Micro F1,Macro F1
    fileTagList=['1TD','2TS','3UD','4US']
    filetag=''
    for ft in range(0,len(fileTagList)):
        if fileTagList[ft] in labeledPath:
            filetag=fileTagList[ft]
            break
    performanceResult.append(filetag)            # 0 文件的来源 看是1TD 还是2TS
    performanceResult.append(Mi_cro_f1)  # 1 不受小样本类别影响的 Micro F1
    performanceResult.append(np.mean(Biglabel_FPR))  # 2 FPR
    performanceResult.append(np.mean(Biglabel_FNR))  # 3 FNR
    performanceResult.append(time.time()-startTime)  #额外加的 训练时间

    csv_Per_result=[]
    # if(t==0):#如果是第一次调用tryMethod函数需要写入列名
    #     csv_Per_result.append(['fileSource','Micro F1','大类0.03的平均FPR','大类0.03的平均FNR'])
    csv_Per_result.append(performanceResult)

    # data_write_csv(pathBigLabel, csv_Per_result)
    return csv_Per_result #直接返回性能的数值


#------------------------------------------------ 调用的主要函数------------------------------------------------------------------------------------------------------
#------------------------------------------------数据读取等------------------------------------------------------------------------------------------------------
def RFModelClass(labeledDataDf,pklPath,featuresMy):
    '''
    :param labeledDataDf:agg得到的已标记的数据 dataframe格式
    :param pklPath: 模型pkl文件的保存路径
    :param featuresMy: 选取的可用特征+label
    :return:csv_Performance #返回性能数值['fileSource','Micro F1','大类0.03的平均FPR','大类0.03的平均FNR']
    '''
    startTime=time.time()

    df = labeledDataDf #打好标签的数据
    pathPKL=pklPath#存放模型的路径
    # pathBigLabel=performancePath#大类的指标等等.

    dfC=df[featuresMy]  # 论文2新特征 版本二

    # 数值归一化方案一 全部都归一化：对hash和速度先归一化，然后乘以100
    # 归一化 equation1  x^=(x-min)/(max-min)
    # chash = (dfC['C_hash'] - dfC['C_hash'].min()) / (dfC['C_hash'].max() - dfC['C_hash'].min())
    # cspd = (dfC['C_Pck_Spd'] - dfC['C_Pck_Spd'].min()) / (dfC['C_Pck_Spd'].max() - dfC['C_Pck_Spd'].min())
    # sspd = (dfC['S_Pck_Spd'] - dfC['S_Pck_Spd'].min()) / (dfC['S_Pck_Spd'].max() - dfC['S_Pck_Spd'].min())
    # 归一化 equation2  x^=(x-Avg)/(Standard Diation)
    if dfC['C_hash'].std() == 0:
        chash = dfC['C_hash'] / 16
    else:
        chash = (dfC['C_hash'] - dfC['C_hash'].mean()) / dfC['C_hash'].std()
    if dfC['C_Pck_Spd'].std() == 0:  # std==0 只有CSpd全为0,不会这样的
        print(dfC['C_Pck_Spd'].values.tolist())
        cspd = dfC['C_Pck_Spd']
    else:
        cspd = (dfC['C_Pck_Spd'] - dfC['C_Pck_Spd'].mean()) / dfC['C_Pck_Spd'].std()
    if dfC['S_Pck_Spd'].std() == 0:  # std==0 只有CSpd全为0,可能服务器为0
        print(dfC['S_Pck_Spd'].values.tolist())
        sspd = dfC['S_Pck_Spd']
    else:
        sspd = (dfC['S_Pck_Spd'] - dfC['S_Pck_Spd'].mean()) / dfC['S_Pck_Spd'].std()

    chash = chash * 100
    cspd = cspd * 100
    sspd = sspd * 100

    # 安全删除，如果用del是永久删除
    dfC2 = dfC.drop(['C_Pck_Spd', 'S_Pck_Spd','C_hash'], axis=1)
    # 把规格化的列插入到数组中,插入最开始 顺序无所谓的，特征只是用来分类的，之后的类别会和原始的全部列结合的，和这里的特征顺序无关的.
    dfC2.insert(0, 'C_hash', chash)
    dfC2.insert(0, 'C_Pck_Spd', cspd)
    dfC2.insert(0, 'S_Pck_Spd', sspd)

    data =dfC2.values
    np.random.shuffle(data) #打乱数据顺序，多维矩阵中只 对第一维（行）做打乱顺序操作

    model_RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators=100) #使用100个弱分类器.

    csv_Performance=try_method(model_RandomForestClassifier,data,df,pklPath,startTime)
    joblib.dump(model_RandomForestClassifier,pathPKL)#保存训练模型 双向流

    return csv_Performance #返回性能数值['fileSource','Micro F1','大类0.03的平均FPR','大类0.03的平均FNR','TrainTime']