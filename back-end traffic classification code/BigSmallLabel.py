# 找出所有流量记录中每一类别的占比
# coding: utf-8

from collections import Counter

#-——————————————————————------------------------数据读取与处理————————————————————————————--------------------------------————————————-----------————————————-----------————————————-----------

def BigSmalllabel(dfOrigin):#返回1 大类的label  2 大类出现的占比
    dfLabel=dfOrigin[['label']]
    #输出每个类别占所有记录的占比
    csv_dataLabel=[]
    for indexs in dfLabel.index:#遍历所有行 将dataframe格式的转换为list[list ]
        csv_dataLabel.append(int(dfLabel.loc[indexs].values[0:]))#values[0:].
    CounterLabel=Counter(csv_dataLabel)

    keys=list(CounterLabel.keys())
    values=list(CounterLabel.values())
    sum=0
    for i in range(0,len(values)):
        sum+=values[i]

    listBigSmallLabel=[]
    for i in range(0,len(keys)):
        row=[]
        row.append(keys[i])
        amount=values[i]/sum
        row.append(amount)
        listBigSmallLabel.append(row)

    BigLabel=[] #记录阈值满足的label
    BigLabelPro=[] #记录对应BigLabel中label的相应的占比
    thred1=100/dfLabel.shape[0] #只有样本数只有几个和几十的才会被错分，就暂且算小类别.
    for i in range(0,len(listBigSmallLabel)):
        row=listBigSmallLabel[i]
        if(row[1]>=thred1):
            BigLabel.append(row[0])
            BigLabelPro.append(row[1])

    return BigLabel,BigLabelPro

# dfOrigin = pd.read_csv("./dataSet/train_AI_aggResult/agg_AI_32_0.05_alldata_79975.csv")
# BigSmallLabel(dfOrigin)
