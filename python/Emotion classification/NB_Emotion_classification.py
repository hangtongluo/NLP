# -*- coding: utf-8 -*-
"""
1.爬取评论（商品）随便2.预处理，就是分词，词性标注jieba就行3训练集送入朴素贝叶斯4 测试，输出好评差评
"""

'''
提示：当出现FileNotFoundError: File b'comment.txt' does not exist，
    不要担心由于页面加密会出现查询不到情况，请重新运行
'''
import os
import re
import pandas as pd
import requests
import jieba
import jieba.posseg as pseg #按词性分词
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, recall_score, f1_score, auc, roc_curve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#———————————————————————————————1、爬取京东评论数据————————————————————————————————————
print('*****************开始爬取评论数据*******************')
if os.path.exists('comment.txt'):
    os.remove('comment.txt')
else:
    f = open('comment.txt','a')

count = 0
#需要按照文档中的寻找网址方法将评论数据页变为两部分，方便进行遍历
#url = 'https://sclub.jd.com/comment/productPageComments.action?callback=\
#        fetchJSON_comment98vv1&productId=6790503&score=0&sortType=5&page='
#url2 = '&pageSize=10&isShadowSku=6560154&rid=0&fold=1'
url = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv2023&productId=6372567&score=3&sortType=5&page='
url2 = '&&pageSize=10&isShadowSku=0&rid=0&fold=1'
num = 100 #需要爬虫的页面数（数据量不要太大，太大运行太慢）
for i in range(0, num):#输入需要爬虫的页面数
    try:
        html = requests.get(url + str(i) + url2)
        html = html.text
       
        #使用正则提取评论信息
        content1 = re.findall(r'"guid".*?,"content":(.*?),',html)
        
        #对提取的评论信息进行去重（由于爬去的数据会出现重复的情况）
        content2=[]
        temp = ''
        for c in content1:
            if temp != c:
                content2.append(c)
            temp = c

        #使用正则提取score字段信息
        score = re.findall(r'"referenceImage".*?,"score":(.*?),',html)
        
        for s,c in zip(score,content2):
            count += 1
            c = c.replace('\\n','')
            f.write(str(count)+'\t' + str(s)+'\t' + c)
            f.write('\n')
    except:
        print('爬取第'+str(i)+'页出现问题')
        print('由于页面加密会出现查询不到情况，请重新运行')
#        continue
        break
f.close()
print('*****************结束爬取评论数据*******************')
#——————————————————————————————1、爬取京东评论数据————————————————————————————————————

#———————————————————————————————2、jiebe分词评论数据————————————————————————————————————
print('*****************开始jiebe分词评论数据*******************')
#导入数据进行分词
data = pd.read_table('comment.txt', header=None, encoding='GBK')
data.columns = ['ind_num', 'score', 'content']

#对用户的情感进行标记（好评为：5星，非好评为：小于5星）（自己定义边界）
data['score'] = data['score'].apply(lambda x: 1 if x==5 else 0)

#对数据中的用户未评价数据进行过滤
data = data[data['content'] != "此用户未填写评价内容"]


#定义结巴分词函数
def jieba_token(sentence):
    sentence = jieba.cut(sentence)   
    return " ".join(sentence)

#对content进行分词
data['content'] = data['content'].apply(lambda x: jieba_token(x))

#保存分词后的文件
data.to_csv('comment_token.txt', index=False, sep='\t')
print('*****************结束jiebe分词评论数据*******************')

print('*****************训练词袋生成词向量*******************')
#得到所有的句子
contents = []
for content in data['content'].values:
    contents.append(content)

#导入中文停用词参考下面的网址（中文停用词库.txt）
#https://github.com/chdd/weibo/tree/master/stopwords
stop_words = pd.read_table('中文停用词库.txt', header=None, encoding='GBK', sep='\\n')
stop_words = list(stop_words[0])

#训练词袋模型
vectorizer = CountVectorizer(min_df=1, encoding='GBK', stop_words=stop_words)
vec = vectorizer.fit(contents)

#分割训练集合测试集(4:1)
length = data.shape[0]
length_flag = int((4/5) * length)

train_X = data['content'][:length_flag].values
test_X = data['content'][length_flag:].values              
train_y = data['score'][:length_flag].values
test_y = data['score'][length_flag:].values

#转换成词袋向量
vec_train_X = vec.transform(list(train_X))            
vec_test_X = vec.transform(list(test_X)) 

print('*****************训练词袋生成词向量*******************')
#——————————————————————————————2、jiebe分词评论数据————————————————————————————————————

#————————————————————————————————3、训练朴素贝叶斯———————————————————————————————————
print('*****************开始训练朴素贝叶斯*******************')
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
nb = MultinomialNB()
nb = nb.fit(vec_train_X, train_y)

train_pre = nb.predict(vec_train_X)
test_pre = nb.predict(vec_test_X)

print('*****************结束训练朴素贝叶斯*******************')
#———————————————————————————————3、训练朴素贝叶斯————————————————————————————————————

#————————————————————————————————4、测试，输出准确率———————————————————————————————————
print('*****************开始测试，输出准确率*******************')
#计算准确率，召回率，f1值
train_accuracy = accuracy_score(train_y, train_pre)
train_recall = recall_score(train_y, train_pre)
train_f1 = f1_score(train_y, train_pre)

test_accuracy = accuracy_score(test_y, test_pre)
test_recall = recall_score(test_y, test_pre)
test_f1 = f1_score(test_y, test_pre)

print('train accuracy：%.4f' % train_accuracy)
print('train recall：%.4f' % train_recall)
print('train f1：%.4f' % train_f1)
print('test accuracy：%.4f' % test_accuracy)
print('test recall：%.4f' % test_recall)
print('test f1：%.4f' % test_f1)


plot_data = pd.DataFrame({'train':[train_accuracy, train_recall, train_f1], 
                          'test':[test_accuracy, test_recall, test_f1]})
plot_data.index = ['准确率', '召回率', 'f1']

#画出测试和训练数据的准确率，召回率，f1值柱状图
plt.figure(figsize=(16,10))
plot_data.plot.barh(cmap=plt.cm.winter)
plt.title('基于朴素贝叶斯的京东用户评论数据情感分析-准确率-召回率-f1图')
plt.savefig('基于朴素贝叶斯的京东用户评论数据情感分析-准确率-召回率-f1图.png')
plt.show()

print('*****************结束测试，输出准确率*******************')
#———————————————————————————————4、测试，输出准确率————————————————————————————————————










