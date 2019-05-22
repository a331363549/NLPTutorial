# -*- coding:utf-8 -*-
# @Time     :2019/4/11 0011 14:13
# @author   :ding

"""
封装对外提供接口
"""

from .corpus import get_corpus
from .model import get_model

__all__ = ["extract_feature", "trian", "tagger"]


def extract_feature():
    """
    抽取语料特征
    """
    corpus = get_corpus()
    corpus.initialize()
    corpus.cal_feature()


def train(epoch=None, show=None):
    """
    训练模型
    """
    model = get_model()
    model.train(epoch=epoch, show=show)


def tagger(sentence):
    """
    词性标注
    """
    model = get_model()
    return model.predict(sentence)
