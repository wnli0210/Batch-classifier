#代码功能：使用已经训练好的模型对测试集数据进行预测
# 1 使用已有模型来预测测试集流量（DDoS+MAWI1.5） 得到预测结果
# 2 对预测结果进行概念漂移检测
#   2.0）首先需要统计三元组有哪些，每个三元组一共有多少条样本在里面
#   2.1）查询测试集中每一条样本的预测概率值p；如果p<0.6就将这些样本归为 漂移样本
#   2.2）将漂移样本按照 三元组 来分析，假设一个三元组中的 漂移样本数/总样本数=Pro>0.9 就说这个三元组一定是发生了异常
#   2.3) 判定为异常三元组所包含的样本数/全部测试集的样本数=Pro2>0.3 就直接触发更新流程

import csv
import numpy as np
import pandas as pd
import time
import joblib
from collections import Counter

#------------------------------------------------模型预测----------------------------------------------------------------
def Predict(pathPKL,dfOrigin, Xtest):
    '''
    :param pathPKL: pkl模型的保存路径
    :param dfOrigin: 原始的特征数据（包含无效特征）
    :param Xtest: 处理后的可用特征
    :return:
       cntDrift, cntAll, proDrifttoAll, canUpdate
       漂移流量的样本数， 流量总样本数,漂移样本比上总样本，是否触发更新
    '''

    rfModel = joblib.load(pathPKL)  # 加载已有分类模型

    # 预测分类标签 带有预测概率值的
    y_Pre_Proba=rfModel.predict_proba(Xtest)

    rowNum = y_Pre_Proba.shape[0]  # shape[0]为矩阵的行号

    #构建 dfCountsID ['SanyuanzuID', 'AllSamples','DriftSamples'] 用于统计“漂移三元组的个数”
    IDList=dfOrigin[['Hash_ID']].values.tolist()
    IDList=[ w for item in IDList for w in item ] #这里将dfValues的二维list转为一维list
    counterDic=Counter(IDList) #统计每个三元组对应的样本数有多少:{(ID1,100),(ID2,98)}
    dfCountsID = pd.DataFrame(counterDic.items(), columns=['SanyuanzuID', 'AllSamples']) #将Counter的计数结果转变为dataframe格式
    dfCountsID["DriftSamples"] = 0   #dfCountsID 中最开始“漂移样本数”为0，后面会根据ID对应的实际样本来统计累加的

    #汇总预测概率小于
    listDriftPro=[] #记录测试集中预测概率低于60%的样本 的实际的预测概率值 最后算AVPPV的
    for i in range(0, rowNum, 1):
        row = []
        row = list(dfOrigin.loc[i].values[0:])  # 取得原始记录的每一行

        # id = row[2]  # local: id为每行的第三个元素 即为下标2
        id = row[5]  # remote: Hash_ID为每行的第6个元素 即为下标5

        preClass_i = np.argmax( y_Pre_Proba[i])  # 每一个样本 中 概率最大的类别

        preValue_i = y_Pre_Proba[i][preClass_i]  # 最大类别对应的预测概率值  or 和最大类别的匹配度

        if (preValue_i < 0.75): # Replay
            #找到三元组ID==当前id的行，将这一行的 漂移样本数 加一
            dfCountsID.loc[(dfCountsID.SanyuanzuID == id),'DriftSamples'] = dfCountsID.loc[(dfCountsID.SanyuanzuID == id),'DriftSamples'] + 1
            #将当前低于阈值0.6的样本 的预测概率值  加入listDriftPro
            listDriftPro.append(preValue_i)

    # 判断每个三元组是不是漂移三元组
    cntDrift=0     #测试集中漂移样本的数量（如果一个三元组是漂移三元组 就把这个三元组的样本数都算作漂移样本）
    cntAll=rowNum  #测试集中所有样本的数量
    for i in range(0,len(dfCountsID)): #dfCountsID ['SanyuanzuID', 'AllSamples','DriftSamples']  最开始漂移样本数为0，后面会统计累加的
        proDriftSanyuzu=dfCountsID.iloc[i,2] / dfCountsID.iloc[i,1]   #即为每个id所在行的 'DriftSamples'/ 'AllSamples'
        if(proDriftSanyuzu >= 0.9):
            cntDrift+=dfCountsID.iloc[i,1]  #如果一个三元组是漂移三元组 就把这个三元组的全部样本数都算作漂移样本

    # 判断漂移三元组中的样本个数 占 全部数据的样本 的占比 能否触发更新流程
    proDrifttoAll=0
    if(cntAll>0):
        proDrifttoAll=cntDrift/cntAll
    canUpdate=0 #是否触发更新 默认不触发
    if(proDrifttoAll >= 0.3):
        canUpdate=1;

    print("本次测试的【漂移流量的样本数】：", cntDrift,"【总样本数】：", cntAll)

    #返回 漂移流量的样本数，流量总样本数,漂移样本比上总样本，是否触发更新 cntDrift, cntAll,proDrifttoAll,canUpdate
    return cntDrift, cntAll,proDrifttoAll,canUpdate

def RFmodelPredict(featureList,dfOrigin,pathPKL):
    '''
    :param featureList:选择的特征集合
    :param dfOrigin: sketch的测试集--原始特征数据集合 dataframe
    :param pathPKL: 存放pkl模型的路径
    :return:
       cntDrift, cntAll, proDrifttoAll, canUpdate
       漂移流量的样本数， 流量总样本数,漂移样本比上总样本，是否触发更新
    '''
    # ----------------------------------------数据读取与预处理------------------------------------------------------------
    dfOrigin = dfOrigin.reset_index(drop=True)

    csv_data = []
    #tcp特征
    featuresMy=featureList
    #['C0', 'Cd', 'Cf', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'S0', 'Sd', 'Sf', 'Sa1', 'Sa2', 'Sa3', 'Sa4', 'C_Pck_Spd', 'S_Pck_Spd', 'C_hash'] tcp
    #['C_hash', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'Sa1', 'Sa2', 'Sa3', 'Sa4', 'C_Pck_Spd', 'S_Pck_Spd'] udp

    dfC = dfOrigin[featuresMy]  # 论文2新特征 版本二
    # 数值归一化方案一 全部都归一化：对hash和速度先归一化，然后乘以100
    # 归一化 equation1  x^=(x-min)/(max-min)
    # chash = (dfC['C_hash'] - dfC['C_hash'].min()) / (dfC['C_hash'].max() - dfC['C_hash'].min())
    # cspd = (dfC['C_Pck_Spd'] - dfC['C_Pck_Spd'].min()) / (dfC['C_Pck_Spd'].max() - dfC['C_Pck_Spd'].min())
    # sspd = (dfC['S_Pck_Spd'] - dfC['S_Pck_Spd'].min()) / (dfC['S_Pck_Spd'].max() - dfC['S_Pck_Spd'].min())
    # 归一化 equation2  x^=(x-Avg)/(Standard Diation)
    if dfC['C_hash'].std()==0:
        chash=dfC['C_hash']/16
    else:
        chash = (dfC['C_hash'] - dfC['C_hash'].mean()) / dfC['C_hash'].std()
    if dfC['C_Pck_Spd'].std()==0: #std==0 只有CSpd全为0,不会这样的
        cspd=dfC['C_Pck_Spd']
    else:
        cspd = (dfC['C_Pck_Spd'] - dfC['C_Pck_Spd'].mean()) / dfC['C_Pck_Spd'].std()
    if dfC['S_Pck_Spd'].std()==0:#std==0 只有CSpd全为0,可能服务器为0
        sspd = dfC['S_Pck_Spd']
    else:
        sspd = (dfC['S_Pck_Spd'] - dfC['S_Pck_Spd'].mean()) / dfC['S_Pck_Spd'].std()

    chash = chash * 100
    cspd = cspd * 100
    sspd = sspd * 100

    # 安全删除，如果用del是永久删除
    dfC2 = dfC.drop(['C_Pck_Spd', 'S_Pck_Spd', 'C_hash'], axis=1)
    # 把规格化的列插入到数组中,插入最开始 顺序无所谓的，特征只是用来分类的，之后的类别会和原始的全部列结合的，和这里的特征顺序无关的.
    dfC2.insert(0, 'C_hash', chash)
    dfC2.insert(0, 'C_Pck_Spd', cspd)
    dfC2.insert(0, 'S_Pck_Spd', sspd)

    dfC2.fillna(0, inplace=True)  # 前面归一化处理，可能会得到0/0=Nan的情况。需要填充为0.

    # dfC2=dfC2.reset_index(drop=True)

    for indexs in dfC2.index:  # 遍历所有行 将dataframe格式的转换为list[list ]
        csv_data.append(dfC2.loc[indexs].values[0:])  # values[0:]去除每一行的所有列上的数值  所有列
    Xtest = np.array(csv_data)

    print('-----------------------------------------------------开始本次概念漂移检测--------------------------------------------' )
    return Predict(pathPKL,dfOrigin,Xtest) #返回的数据可写入mysql表格

# if __name__ == '__main__':
#     # ----------------------------------------数据读取与预处理------------------------------------------------------------
#     csv_data = []
#     # #tcp特征
#     # featuresMy=['C0', 'Cd', 'Cf', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'S0', 'Sd', 'Sf', 'Sa1', 'Sa2', 'Sa3', 'Sa4', 'C_Pck_Spd', 'S_Pck_Spd', 'C_hash']
#     #udp的特征
#     featuresMy=['C_hash', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'Sa1', 'Sa2', 'Sa3', 'Sa4', 'C_Pck_Spd', 'S_Pck_Spd']
#
#     filetag='4US' # 1TD 2TS 3UD 4US
#     dfOrigin = pd.read_csv('./dataSet/newSketchPool/test/'+filetag+'.csv', encoding='gbk')  # sketch的测试集
#     pathPKL = './model/oldSketch/model_'+filetag+'.pkl'  # 存放模型的路径
#
#     cntDrift, cntAll, proDrifttoAll, canUpdate=RFmodelPredict(featuresMy,dfOrigin,pathPKL) #返回的数据可写入mysql
