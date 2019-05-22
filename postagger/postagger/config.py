# -*- coding:utf-8 -*-
# @Time     :2019/4/11 0011 14:16
# @author   :ding

"""
配置封装
"""

import configparser

__config = None

def get_config(config_file_path="postagger/conf/config.conf"):
    """
    单例配置获取
    :param config_file_path: 配置文件路径
    :return: 配置信息
    """
    global __config
    if not __config:
        config = configparser.ConfigParser()
        config.read(config_file_path)
    else:
        config = __config
    return config


