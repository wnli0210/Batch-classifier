#!/usr/bin/env python
# coding: utf-8
'''
  1 完成分块聚类，为数据打上标签： agglomerativeClass(dataPath,ChunkSizeMy,resultPath1,resultPath2,featuresMy,colNameMy)
  2 对已标记数据放入Rf中训练，得到Rf模型： 调用公共类RFmodelTranClass.py的RFModelClass(labeledDataDf,pklPath,featuresMy)
  3 参数设置在main中 后续其他函数调用本py文件，设置参数就参考main中的
'''

import csv
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
from RFmodelTrainClass import RFModelClass
import joblib
from config import Config
from mysql_writer import MySQLWriter

#******************************************************************AGG 聚类 的部分****************************************************************************
# -——————————————————————------------data_write_csv用于写入文件————————————————————————————--------------------------------————————————-----------————————————-----------————————————-----------
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        for data in datas:
            writer.writerow(data)  # writerow()方法是一行一行写入.
    # print("保存文件成功，处理结束")

# -——————————————————————------------对训练后的 一个块 取出每一个类的流并Merge————————————————————————————---------
def ClassInBlock_Merge(per_chunk_Result, chunk_labelNum):
    # per_chunk_Result对应于csv_chunk_Result，为一个块中全部流量的分类结果（原始数据+label）
    # chunk_labelNum为当前块的总共的分类数
    # 返回结果是一类只有一条流量特征+label
    Merge_Result = []  # 记录块中所有类的合并结果 每一个元素是一个类合并的结果

    for i in range(0, chunk_labelNum, 1):  # 对每一个类的所有流进行聚类, i就是label
        list_Clu = []  # 存放第i类的所有流
        result = []  # 存放每一个类聚类后的结果

        for j in range(0, len(per_chunk_Result), 1):
            row = per_chunk_Result[j]  # 当前的一条流
            label_flow = row[len(row) - 1]
            if (i == label_flow):  # 这条流的标签等于当前需要合并的类的类序号
                list_Clu.append(row)

        result = Merge(list_Clu)  # 返回的新分类节点（平均特征 + label）
        Merge_Result.append(result)

    return Merge_Result

# —————————————————————————Merge模块用于合并 一个类 中的多个流 得到一条新流量特征—————————
def Merge(listClu):  # 输入是完整的原始特征+label 的多行记录
    length = len(listClu[0]) - 1
    label = listClu[0][length]  # 记住这一类的类标号

    # #tcp 中 featuresMy在colNameMy的第8列到倒数第二列   下标为7:length  (len-1=length这个位置到不了)
    # featuresMy = ['C0', 'Cd', 'Cf', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'S0', 'Sd', 'Sf', 'Sa1', 'Sa2', 'Sa3', 'Sa4','C_Pck_Spd', 'S_Pck_Spd', 'C_hash']
    # colNameMy = ['Redis_Key', 'Ratio', 'Protocol', 'IP', 'Port', 'Hash_ID', 'Time', 'S0', 'Sa1', 'Sa2', 'Sa3', 'Sa4',
    #              'C0','Ca1', 'Ca2', 'Ca3', 'Ca4', 'C_hash', 'S_Pck_Spd', 'C_Pck_Spd', 'Sf', 'Cf', 'Sd', 'Cd', 'label']  # tcp

    # udp 中featuresMy在colNameMy的9-12、14-20列 下标为8：12 以及 13：20
    # featuresMy = ['C_hash', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'Sa1', 'Sa2', 'Sa3', 'Sa4', 'C_Pck_Spd', 'S_Pck_Spd']
    # colNameMy = ['Redis_Key', 'Ratio', 'Protocol', 'IP', 'Port', 'Hash_ID', 'Time', 'S0', 'Sa1', 'Sa2', 'Sa3', 'Sa4',
    #              'C0', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'C_hash', 'S_Pck_Spd', 'C_Pck_Spd', 'Sf', 'Cf', 'Sd', 'Cd', 'label']  # udp

    # list_Clu = [x[7:length] for x in listClu]  # tcp 系统读出设计的特征从第8列开始，到倒数第二列结束    下标为7:length  (len-1=length这个位置到不了)
    list_Clu = [x[8:12]+x[13:20] for x in listClu]  # udp 的featuresMy在colNameMy的9-12、14-20列 下标为8：12 以及 13：20

    array_Clu = np.array(list_Clu)  # list转为numpy数组
    data = array_Clu.astype(np.float32)  # 元素string转为float.
    Avg_Cluster = data.mean(axis=0)  # 按照列求平均值，作为当前类的代表

    list_result = list(Avg_Cluster)
    list_result.append(label)

    return list_result  # 返回的新分类节点（平均特征 + label）

# ------------------------------------normalize ndarray--------------------------------------------------------------------------------------------------
def normalize_func(minVal, maxVal, newMinValue=0, newMaxValue=1):
    def normalizeFunc(x):
        r = (x - minVal) * newMaxValue / (maxVal - minVal) + newMinValue
        return r
    return np.frompyfunc(normalizeFunc, 1, 1)

#----------------------------------聚类的主要函数 调用这个主要函数--------------------------------------------------------------------------------------------
def agglomerativeClass(dataPath,ChunkSizeMy,resultPath1,resultPath2,featuresMy,colNameMy):
    '''
    :param dataPath: 原始特征数据的路径
    :param ChunkSizeMy:块大小
    :param resultPath1: 不分块时 的结果保存路径
    :param resultPath2: 分块时 的结果保存路径
    :param featuresMy: 可用的特征
    :param colNameMy：最后标记好数据的列名
    :return: csv_file #为了方便后面RF 模型的训练，直接返回带标签label的原始数据
    '''

    # ——————————————————————————————————----------------聚类主函数------————————————————-------------————————————————-------------————————————————-------------————————————————-----------------------------------------------------------------------------------------
    # -——————————————————————------------------------数据读取与处理————————————————————————————---------------------------------------------------
    chunkS =ChunkSizeMy   # 论文2检测异常 固定块大小即可
    dfOrigin = pd.read_csv(dataPath,chunksize=chunkS)  # 按块读取原始数据，每块200条流  最初的训练集特征存在mysql里就好
    # df1=dfOrigin.get_chunk(chunkS) #从上到下分块读取文件，每次读取chunks行/一个块chunk
    path1=resultPath1#块数等于1的情况，存放聚类结果的路径
    path2=resultPath2#块数大于1的情况，存放聚类结果的路径
    model = AgglomerativeClustering(n_clusters=None, linkage="average", affinity="cosine", distance_threshold=0.05)
    count = 0  # 块的序号

    # -——————————————————————------------------------将每一块中的数据agg聚类得到各块的聚类结果———————————————————————--------------————---------------
    csv_Results = []  # 存放所有块的agg结果 每一个元素是一个csv_chunck_Result
    label_Results = []  # 存放所有块的agg结果对于的label数量  每一个元素是一个块的label最大标号
    count = 0  # 记录块的数量
    list_feature=featuresMy

    for chunk in dfOrigin:  # 对每一个块使用Agg模型来进行标签标记
        dfC = chunk[list_feature]  # tcp

        # 数值归一化方案一 全部都归一化：对hash和速度先归一化，然后乘以100
        chash = (dfC['C_hash'] - dfC['C_hash'].min()) / (dfC['C_hash'].max() - dfC['C_hash'].min())
        chash = chash * 100
        cspd = (dfC['C_Pck_Spd'] - dfC['C_Pck_Spd'].min()) / (dfC['C_Pck_Spd'].max() - dfC['C_Pck_Spd'].min())
        cspd = cspd * 100
        sspd = (dfC['S_Pck_Spd'] - dfC['S_Pck_Spd'].min()) / (dfC['S_Pck_Spd'].max() - dfC['S_Pck_Spd'].min())
        sspd = sspd * 100
        # 安全删除，如果用del是永久删除
        dfC2 = dfC.drop(['C_Pck_Spd', 'S_Pck_Spd', 'C_hash'], axis=1)
        # 把规格化的列插入到数组中,插入最开始 顺序无所谓的，特征只是用来分类的，之后的类别会和原始的全部列结合的，和这里的特征顺序无关的.
        dfC2.insert(0, 'C_hash', chash)
        dfC2.insert(0, 'C_Pck_Spd', cspd)
        dfC2.insert(0, 'S_Pck_Spd', sspd)

        dfC2.fillna(0, inplace=True)  # 前面归一化处理，可能会得到0/0=Nan的情况。需要填充为0.

        #df转换为numpy.array
        csv_dataC = []  # 将dfC转为list[list]存储在csv_dataC中
        for index in dfC2.index:
            csv_dataC.append(dfC2.loc[index].values[0:])
        Xc = np.array(csv_dataC,dtype='float')  # 将 list 转换为 numpy.array


        bestlabels = model.fit_predict(Xc)  # 使用Xc训练模型

        Cnt = Counter(bestlabels)  # 计数当前块中流量一共分了多少类
        label_Results.append(len(Cnt))  # 将类别数存入label_Resultd.

        csv_chunk_Result = []  # 存放csv文件中每一块的分类情况
        for index in dfC.index:  # chunk每个只有ChunkSize这么大，但各个chunk的index是连起来的
            # 按顺序存放好每一条原始信息以及对应的label.
            row = []
            row = list(chunk.loc[index].values[0:])  # 取得原始记录的每一行数据
            index_location = index - chunkS * count
            row.append(bestlabels[index_location])
            csv_chunk_Result.append(row)

        csv_Results.append(csv_chunk_Result)  # 将每一块的聚类结果作为一个整体 放入csv_Results中
        print("---------------第" + str(count) + "块分类结束--------------------")
        count = count + 1  # 快序号加一，处理下一个块
    print("---------------第" + str(count) + "块分类结束（最后一块可能不足）--------------------")

    colName=colNameMy

    if(1==count):#-------------------------------------------情况一：如果数量量只够分一块 直接将上面的聚类结果写入文件即可--------------------------------------------------------
        csv_file_0=[]
        column_0=colName;
        csv_file_0.append(column_0)
        for i in range(0,len(csv_Results)):
            csv_file_0=csv_file_0+csv_Results[i]
        data_write_csv(path1,  csv_file_0)

    else:#--------------------------------------------------情况二：如果分块不只1块，才需要合并块--------------------------------------------------------------------------
        # -——————————————————————------------------------对所有块中的聚类结果进行合并———————————————————————————-------------------------------------------------------------
        MergeClass_Chunk = []  # 记录所有块数据按类合并得到的数据 每一个元素为C_Merge_Res.
        for i in range(0, len(csv_Results), 1):
            perChunk_Result = csv_Results[i]  # csv_chunk_Result 将每一块的聚类结果作为一个整体 放入csv_Results中
            perChunk_LabelNum = label_Results[i]
            c_Merge_Res = ClassInBlock_Merge(perChunk_Result, perChunk_LabelNum)  # 返回的是一类只有一行（14个平均特征加上label）,每块中有n类就有n行合并后的类特征+label.
            MergeClass_Chunk.append(c_Merge_Res)
        #-——————————————————————------------------------对合并过的每一块数据 再用Agg层次聚类———————————————————————————-------------------------------------------------------------
        Data_afterMerge=[]#记录所有块中的所有类特征 作为agg的输入数据
        label_Chunk=[]#总的块的label变化 =[label_PerChu,label_PerChu，..]
        for i in range(0,len(MergeClass_Chunk),1):
            label_PerChu=[]#记录每一块的label变化

            c_Mer=MergeClass_Chunk[i] #每一块的类合并总的结果.n类n行，每行为特征+label.

            for j in range(0,len(c_Mer),1):
                row=c_Mer[j]#一行类合并信息=特征+label.
                label_PerChu.append(row[len(row)-1])#将每行类合并信息的label按顺序放入label_Perchu
            label_Chunk.append(label_PerChu)  # 每一块的label信息 放入label_Chunk.

            c_Mer_fetures= [x[0:len(x)-1] for x in c_Mer] #取出其中的特征0~len-2 ,最后为label.
            Data_afterMerge=Data_afterMerge+c_Mer_fetures #每一块的类合并的特征 汇总到Data_afterMerge.

        # data_write_csv('./result/Input.csv',Data_afterMerge)

        # 这里的特征是新的分类特征,从原始的data中取来的原始数据 也需要归一化
        XD_aft_Merge = np.array(Data_afterMerge, dtype='float32')  # 将 list 转换为 numpy.array
        # 这里XD_aft_Merge中每一行：[C0, Cd, Cf, Ca1,Ca2,Ca3,Ca4,S0,Sd,Sf,Sa1,Sa2,Sa3,Sa4,C_Pck_Spd,S_Pck_Spd,C_hash]
        df_aft_merge = pd.DataFrame(XD_aft_Merge)
        df_aft_merge.columns = list_feature;  # [C0, Cd, Cf, Ca1,Ca2,Ca3,Ca4,S0,Sd,Sf,Sa1,Sa2,Sa3,Sa4,C_Pck_Spd,S_Pck_Spd,C_hash]

        # 归一化方案一 改hash和速度：先归一化然后再乘以100
        chash = (df_aft_merge['C_hash'] - df_aft_merge['C_hash'].min()) / (df_aft_merge['C_hash'].max() - df_aft_merge['C_hash'].min())
        chash = chash * 100
        cspd = (df_aft_merge['C_Pck_Spd'] - df_aft_merge['C_Pck_Spd'].min()) / (df_aft_merge['C_Pck_Spd'].max() - df_aft_merge['C_Pck_Spd'].min())
        cspd = cspd * 100
        sspd = (df_aft_merge['S_Pck_Spd'] - df_aft_merge['S_Pck_Spd'].min()) / (df_aft_merge['S_Pck_Spd'].max() - df_aft_merge['S_Pck_Spd'].min())
        sspd = sspd * 100
        # 安全删除，如果用del是永久删除
        dfC2 = df_aft_merge.drop(['C_Pck_Spd', 'S_Pck_Spd','C_hash'], axis=1)
        # 把规格化的列插入到数组中,插入最开始 顺序无所谓的，特征只是用来分类的，之后的类别会和原始的全部列结合的，和这里的特征顺序无关的.
        dfC2.insert(0, 'C_hash', chash)
        dfC2.insert(0, 'C_Pck_Spd', cspd)
        dfC2.insert(0, 'S_Pck_Spd', sspd)

        # df转换为numpy.array
        csv_dataC = []  # 将dfC转为list[list]存储在csv_dataC中
        for index in dfC2.index:
            csv_dataC.append(dfC2.loc[index].values[0:])
        XD_aft_Merge1 = np.array(csv_dataC, dtype='float')  # 将 list 转换为 numpy.array

        bestlabels_aft_Merge = model.fit_predict(XD_aft_Merge1)  # 使用XD训练模型

        #bestlabels为之前所有快合并特征的全部标记 有序的，需要按照每块合并信息的长度 对bestlabels做切分，将后面的label对应之前的 合并流label
        Chunk_bestlabels=[] #记录所有块的对应的agg的label记录，每一个元素为每一块合并流 对应的agglabel
        chunk_StartNum=0 #记住每块的开始位置
        for i in range(0,len(label_Chunk)):
            Perchu_bestLabels=[] # 每一块合并流 对应的agglabel

            label_PerChu=label_Chunk[i] #每一块的合并流的标签记录
            lenChunk=len(label_PerChu) #每一块合并流的长度

            Perchu_bestLabels=bestlabels_aft_Merge[chunk_StartNum:chunk_StartNum+lenChunk]

            chunk_StartNum=chunk_StartNum+lenChunk #！！

            Chunk_bestlabels.append(Perchu_bestLabels)#将每一块对应的agg的label加入Chunk_bestLabel

        # label_Chunk记录所有块的合并后的label,每一个元素为一个块的label记录list  如chunk1原始有23类=[0，1，...,22]=before_label
        # Chunk_bestlabels记录所有块的合并流量的agg得label,,每一个元素为一个块的合并流量的label记录list  如chunk1的合并类流量agg后分为后面的5类=[0，1，6，9，12]=after_label
        # 用labelInChunk_OtoF 记录所有块的 合并后label-》agg的label的映射 每一个元素row=[before_label,after_label]
        labelInChunk_OtoF=[] #记录所有块的 合并后label-》agg的label的映射
        for i in range(0,len(label_Chunk)):
            row=[]
            row.append(label_Chunk[i])# label_Chunk[i]：如chunk1原始有23类=[0，1，...,22]
            row.append(Chunk_bestlabels[i])# Chunk_bestlabels[i]：如chunk1的合并类流量agg后分为后面的5类=[0，1，6，9，12]
            labelInChunk_OtoF.append(row)

        # -----------------------------------原始的agg结果中label的替换-----------------------------------------------------------
        for i in range(0, len(csv_Results), 1):
            perChunk_csv_Result = csv_Results[i]  # 第i块数据的原始流的agg结果.
            label_OtoF = labelInChunk_OtoF[i]  # 第i块数据的label映射记录=[before_label,after_label]
            beforeLabels = label_OtoF[0]
            afterLabels = label_OtoF[1]
            for j in range(0, len(perChunk_csv_Result), 1):
                row = perChunk_csv_Result[j]  # 每一行为原始数据+原始的label
                labelInRow = row[len(row) - 1]
                indexInLabels = beforeLabels.index(labelInRow)  # 找到当前行中label在合并类labels中的索引.
                row[len(row) - 1] = afterLabels[indexInLabels]  # 将对应下标位置的 最后的label，替换到这一行的label中
                perChunk_csv_Result[j] = row  # 将换了标记的行换回到块中.

        # csv_Results中所有块的agg结果 已经 替换好label
        csv_file = []

        column=colName
        csv_file.append(column)
        for i in range(0, len(csv_Results), 1):  # 三维的list变为二维的list
            csv_file = csv_file + csv_Results[i]
        data_write_csv(path2, csv_file)

        return csv_file #为了方便后面RF 模型的训练，直接返回带标签label的原始数据

    # timeBCAC=time.time() - start_time
    # row=[]
    # row.append(chunkS)
    # row.append(count)
    # row.append(timeBCAC)
    # fileResult=[]
    # fileResult.append(row)
    # data_write_csv('./result/periodical/time.csv',fileResult)
    # # 记录程序运行的时间
    # print("一共用时为：",time.time() - start_time,"秒.")  # 记录程序运行的时间

if __name__ == '__main__':
    filtTag='3UD' #4US
    dataPath1 = "./dataSet/newSketchPool/train/"+filtTag+".csv"  #聚类原材料：sketch出来的原特征文件
    ChunkSizeMy1 = 2000       #块大小 tcp:20000  udp:2000
    resultPath11 = './result/'+filtTag+'_agg_labeled.csv'  #不分块时 的聚类完成的数据的保存路径
    resultPath21 = './result/'+filtTag+'_agg_labeled.csv'  #分块时 的聚类完成的数据的保存路径  可以和上面一样，反正块大小固定了，不用讨论块大小了

    # #tcp
    # featuresMy = ['C0', 'Cd', 'Cf', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'S0', 'Sd', 'Sf', 'Sa1', 'Sa2', 'Sa3', 'Sa4','C_Pck_Spd', 'S_Pck_Spd', 'C_hash']
    # colNameMy = ['Redis_Key', 'Ratio', 'Protocol', 'IP', 'Port', 'Hash_ID', 'Time', 'S0', 'Sa1', 'Sa2', 'Sa3', 'Sa4',
    #              'C0','Ca1', 'Ca2', 'Ca3', 'Ca4', 'C_hash', 'S_Pck_Spd', 'C_Pck_Spd', 'Sf', 'Cf', 'Sd', 'Cd', 'label']  # tcp
    # features_RFTrain_My = ['C0', 'Cd', 'Cf', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'S0', 'Sd', 'Sf', 'Sa1', 'Sa2', 'Sa3', 'Sa4','C_Pck_Spd', 'S_Pck_Spd', 'C_hash','label']

    # udp
    featuresMy=['C_hash', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'Sa1', 'Sa2', 'Sa3', 'Sa4', 'C_Pck_Spd', 'S_Pck_Spd']
    colNameMy=['Redis_Key', 'Ratio', 'Protocol', 'IP', 'Port', 'Hash_ID', 'Time', 'S0', 'Sa1', 'Sa2', 'Sa3', 'Sa4',
               'C0','Ca1', 'Ca2', 'Ca3', 'Ca4', 'C_hash', 'S_Pck_Spd', 'C_Pck_Spd', 'Sf', 'Cf', 'Sd', 'Cd', 'label'] #udp
    features_RFTrain_My=['C_hash', 'Ca1', 'Ca2', 'Ca3', 'Ca4', 'Sa1', 'Sa2', 'Sa3', 'Sa4', 'C_Pck_Spd', 'S_Pck_Spd','label']

    print("|----------------------------------------启动聚类主函数！（本次聚类流量信息：type="+filtTag+"; blockSize=2,000）----------------------------------------|")
    labeledDataList=agglomerativeClass(dataPath1, ChunkSizeMy1, resultPath11, resultPath21,featuresMy,colNameMy)  #调用聚类函数 完成聚类  返回的是聚类完成的list格式的原始特征数据+label
    print("|----------------------------------------批量聚类已完成！（本次聚类流量信息：type="+filtTag+"; blockSize=2,000）----------------------------------------|")
    print()

    labeledDataDf=pd.DataFrame(labeledDataList)
    labeledDataDf=labeledDataDf.drop(index=[0])  #删除df的第一行  为之前的列名，这里被当作数据值了
    labeledDataDf.columns=colNameMy #为df增加列名 方便抽取某些列的数据

    pklPath1 = './model/'+filtTag+'_rfmodel_new.pkl'  #pkl模型文件 的保存路径
    print("|----------------------------------------------------------------开始构建RF Model-------------------------------------------------------------------|")
    performce=RFModelClass(labeledDataDf,pklPath1,features_RFTrain_My)  #返回性能数值['fileSource','Micro F1','大类0.03的平均FPR','大类0.03的平均FNR','TrainTime']
    print(performce)
    # 实例化配置读取模块
    cfg = Config()
    # 实例化MySQL结果写入模块
    mysql_writer = MySQLWriter(cfg=cfg)
    mysql_writer.write(cfg.MYSQL_TABLE_NAME2,
                       cfg.MYSQL_TABLE_ITEMS2,
                       performce,
                       cfg.MYSQL_TABLE_ITEMS_NUM2)  # 结果写入MySQL的表3
    print("|-----------------------------------------------------------RF Model已成功保存为pkl文件！-------------------------------------------------------------|")





