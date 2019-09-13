from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

def datasets_demo():
    """
    sklearn数据集使用
    :return;
    """
    #获取数据集
    # iris=load_iris()
    # iris['DESCR']
    # print('鸢尾花数据集:',iris)
    # print('查看数据集描述:',iris['DESCR'])
    # print('查看特征值的名字:',iris.feature_names)
    # print('查看特征值:', iris.data,iris.data.shape)
    #
    # #数据集划分
    # x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target)
    # print('训练集的特征值:',x_train,x_train.shape)
def dict_demo():
    """
    字典特征抽取
    :return:
    """
    # data=[{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    # #1.实例化一个转换器类
    # trasfer=DictVectorizer(sparse=False)
    # #2.调用fit_transform()
    # data_new=trasfer.fit_transform(data)
    # print('data_new:',data_new)
    # print('特征名字:',trasfer.get_feature_names())
def cut_word(text):
    """
    中文分词：我爱北京天安门 ---》'我 爱 北京 天安门'
    :param text:
    :return:
    """
    return  " ".join(list(jieba.cut(text)))


def count_demo():
    """
    文本特征抽取CountVectorizer
    :return:
    """
    data=['life is short,i like like python','life is too long,i diskile python']

    # 1.实例化一个转换器类
    transfer=CountVectorizer(stop_words=['is','too'])

    # 2.调用fit_transform
    data_new=transfer.fit_transform(data)

    print('data_new:',data_new.toarray())

    print('特征名字:',transfer.get_feature_names())

    return None

def count_chinese_demo2():
    """
    中文文本特征抽取，自动分词
    :return:
    """
    #1.将中文文本进行分词
    data=['一种还是一种今天很残酷，明天更残酷，后天很美好',
          '但是绝对大多数人都死在明天晚上，所以不要放弃今天',
          '我是一种筱筱的排钟，几百万']
    data_new=[]
    for sent in data:
        data_new.append(cut_word(sent))
    #实例化一个转换器
    transfer=CountVectorizer()
    #调用fit_transform
    data_final=transfer.fit_transform(data_new)
    print('data_new:',data_final.toarray())
    print('特征名字:',transfer.get_feature_names())

def tfidf_demo():
    """
    TFIDF的方法进行特征抽取
    :return:
    """
    data = ['一种还是一种今天很残酷，明天更残酷，后天很美好',
            '但是绝对大多数人都死在明天晚上，所以不要放弃今天',
            '我是一种筱筱的排钟，几百万']
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # 实例化一个转换器
    transfer = TfidfVectorizer(stop_words=['一种','所以'])
    # 调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print('data_new:', data_final.toarray())
    print('特征名字:', transfer.get_feature_names())

def minmax_demo():
    """归一化"""
    #1、获取数据
    data=pd.read_csv('data.txt')

    data=data.iloc[:,:3]
    print('data:', data)
    #2、实例化一个转换器
    transfer=MinMaxScaler(feature_range=[2,3]) #可以自己设定
    #3、调用fit——transfrom进行转换
    data_new=transfer.fit_transform(data)
    print('data_new:\n',data_new)

def stand_demo():
    """
    标准化
    :return:
    """
    data=pd.read_csv('data.txt')
    transfer=StandardScaler()
    data=transfer.fit_transform(data)
    print('data:\n',data)

def variance_demo():
    """低方差特征过滤"""

    #1、获取数据
    data=pd.read_csv('data_selection.csv')
    print(data)

    # #2、实例化一个转换器类
    #transfer=VarianceThreshold(threshold=5)#根据需求设置阈值
    # transform=VarianceThreshold

    # #3、调用fit——transfrom
    # data_new = transfer.fit_transform()
    #
    # #计算某两个系数的相关系数
    # r=pearsonr(data[''],data=[''])
    # print('相关系数:'r)

def pca_demo():
    """pca降维小例子"""
    data=[[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    #实例化转换器类,降成2个特征
    transfer=PCA(n_components=0.7)
    #调用fit_transform
    data_new=transfer.fit_transform(data)
    print('data_new:\n',data_new)



if __name__=='__main__':
     datasets_demo()
     # dict_demo()
     # count_demo()
     #count_chinese_demo2()  #出现次数大对我们最终结果产生较大影响
    #tfidf_demo()
    #minmax_demo()
     #stand_demo()
     #variance_demo()
     #ca_demo()
