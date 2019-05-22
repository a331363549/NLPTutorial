# -*- coding:utf-8 -*-
# @Time     :2019/4/11 0011 16:15
# @author   :ding

"""
"""

import sys
from .postagger.api import *


def manage():
    arg = sys.argv[1]
    print(arg)
    if arg == 'trian':
        trian()
    elif arg == 'extract':
        extract_feature()
    else:
        print('Args mus in ["extract","train"].')
    sys.exit()


if __name__ == '__main__':
    trian()
