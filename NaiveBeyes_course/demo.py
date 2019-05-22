import time
import os
import jieba
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


def preprocess(path):
    """文本分词处理"""
    text_with_space = ""
    textfile = open(path, 'r', encoding="utf8").read()
    textcute = jieba.cut(textfile)
    for word in textcute:
        text_with_space += word + ''
    return text_with_space


def loadtrainset(path, classtag):
    allfiles = os.listdir(path)
    processed_textset = []
    allclasstags = []
    for thisfile in allfiles:
        print(thisfile)
        path_name = path + "/" + thisfile
        processed_textset.append(preprocess(path_name))
        allclasstags.append(classtag)
    return processed_textset, allclasstags


# 数据集处理 得到训练集、标签
processed_textdata1, class1 = loadtrainset("./dataset/train/hotel", "宾馆")
processed_textdata2, class2 = loadtrainset("./dataset/train/travel", "旅游")
trian_data = processed_textdata1 + processed_textdata2
classtags_list = class1 + class2

#
count_vector = CountVectorizer()
vector_matrix = count_vector.fit_transform(trian_data)

# TF-IDF
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vector_matrix)

clf = MultinomialNB().fit(train_tfidf, classtags_list)

# 测试集
testset = []

path = "./dataset/test/hotel"
allfiles = os.listdir(path)
hotel = 0
travel = 0

for thisfile in allfiles:
    path_name = path + "/" + thisfile
    new_count_vector = count_vector.transform([preprocess(path_name)])
    new_tfidf = TfidfTransformer(use_idf=False).fit_transform(new_count_vector)

    predict_result = clf.predict(new_tfidf)
    print(''.join(predict_result) + " " + thisfile)
