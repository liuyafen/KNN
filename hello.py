import numpy as np
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
def createDataSet():
    '''创建数据集'''
    group=np.array([[1,101],[5,89],[108,5],[115,8]]) #四组数据集 ,矩阵里的第一个数据代表打斗镜头、第二个接吻镜头
    labels=['爱情片','爱情片','动作片','动作片'] #四组数据集对应的分类标签
    return group,labels     #发挥数据集和分类标签

def classify0(inX, dataSet, labels, k):
    '''knn算法实现
    inX:数据集
    dataSet:训练集
    labels:分类标签
    '''
    dataSetSize=dataSet.shape[0] #返回数据集的行数 ,返回4
    #应用欧氏距离
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet #将intX变成4行，与dataSet相减
    sqDiffMat=diffMat**2  #二维特征相减后平方
    sqDistances=sqDiffMat.sum(axis=1) #平方之后再相加
    distances=sqDistances**0.5 #再开方
    sortedDistIndices=distances.argsort()  #对元素从小到大排列，提取其对应的索引，与K值比较
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]  #提取前k个元素的类别
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1  #get()方法返回键的值，如果值
        #不存在，返回默认值，计算类别次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #classCount字典分解为元组列表,operator.itemgetter(1)根据字典的值进行排序，operator.itemgetter(0)根据字典的键进行排序
    #reverse进行降序排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    '''打开并解析文件，对数据进行分类:
    1 代表不喜欢
    2 代表魅力一般
    3 极具魅力
    返回特征矩阵和分类label向量
    '''
    fr=open(filename) #打开文件 每年获得的飞行常客里程数、玩视频游戏所消耗的时间百分比、每周消费的冰淇淋数
    arrayOLines=fr.readlines() #按行读取文件内容
    numberOfLines=len(arrayOLines) #得到文件的行数
    returnMat=np.zeros((numberOfLines,3)) #解析完成的数据：numberOfLines行，3列
    classLabelVector=[] #返回分类标签向量
    index=0 #行的索引值
    for line in arrayOLines:
        line = line.strip()  #strip剔除空白符
        listFromLine=line.split('\t') #切片
        returnMat[index,:]=listFromLine[0:3] #前3列提取出来，存放在returnMat中
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1]== 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1]== 'largeDoses':
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector
def autoNorm(dataSet):
    '''对数据进行归一化
    return：归一化的特征矩阵
    ranges：数据范围
    minvals：数据最小值
    '''
    minVals=dataSet.min(0)  #获得数据的最小值
    maxvals=dataSet.max(0)   #获得数据的最大值
    ranges=maxvals-minVals
    normDataSet=np.zeros(np.shape(dataSet)) #返回dataset的矩阵行列数
    m=dataSet.shape[0] #返回dataSet的行数
    normDataSet = dataSet - np.tile(minVals, (m, 1)) #原始值减去最小值
    normDataSet = normDataSet / np.tile(ranges, (m, 1)) #除以最大和最小值的差,得到归一化数据
    return normDataSet,ranges,minVals #返回归一化数据结果,数据范围,最小值

def datingClassTest():
    '''分类器测试函数，
    返回：
     normDataSet：归一化后的特征矩阵
    ranges：数据范围
    minvals：数据最小值
    '''
    filename='dict.txt' #打开文件名
    #将返回的特征矩阵和分类向量分别存储到datingDataMat,datingLabels中
    datingDataMat,datingLabels=file2matrix(filename)
    hoRatio=0.10 #取所有数据的10%
    #返回归一化之后的矩阵，数据范围和数据最小值
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0] #返回normMat的行数
    numTestVecs=int(m*hoRatio)#10%测试数据的个数
    errorCount=0.0  #分类错误计数
    for i in range(numTestVecs):
        # 前numTestVecs是个数据作为测试集，后m-numTestVecs个数据作为训练集
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print('分类结果:%d\t真实类别:%d'%(classifierResult,datingLabels[i]))
        if classifierResult!=datingLabels[i]:
            errorCount+=1.0
    print('错误率:%f%%'%(errorCount/float(numTestVecs)*100))

def classifyPerson():
    '''使用算法，通过输入一个人的三维特征，进行分类输出'''

    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #打开的文件名
    filename = "dict.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))



def showdatas(datingDataMat,datingLables):  #参数：特征矩阵和分类label
    '''可数据视化'''
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14) #设置汉字格式
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8)) #设置画布区域样式
    numberOfLabels=len(datingLabels)
    labelscolors=[]
    for i in datingLabels:
        if i==1:   #如果分类标签是1，用黑色代表didntLike
            labelscolors.append('black')
        if i==2:
            labelscolors.append('orange')
        if i==3:
            labelscolors.append('red')
        # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=labelscolors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=labelscolors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=labelscolors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    #图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')

    #添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

    #显示图例
    plt.show()

if __name__=='__main__':
    # group, labels = createDataSet()
    # test=[101,1]  #根据数据预测是什么片子，第一个数据是打斗镜头，第二个是接吻镜头
    # test_class=classify0(test,group,labels,3)
    # print(test_class)


    filename='dict.txt'  #打开文件名
    datingDataMat,datingLabels=file2matrix(filename) #打开并处理数据
    normDataSet, ranges, minVals=autoNorm(datingDataMat)
    print(datingDataMat)
    print(datingLabels)



    #showdatas(datingDataMat,datingLabels)
    print(normDataSet)
    print(range)
    print(minVals)


    datingClassTest()


    classifyPerson()




































