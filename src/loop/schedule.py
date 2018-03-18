import pandas as pd
from pandas import DataFrame
from xlwt import Workbook
# import numpy as np
import random
import matplotlib.pyplot as plt

from src.loop.get_scheduling_DNN import schedule_DNN
from src.loop.get_scheduling_matrix import schedule_matrix
from src.loop.get_scheduling_real import schedule_real


# 输入是矩阵
if __name__ == '__main__':
    # 9个比例循环实现
    # schedule_real()
    for i in range(9):

        schedule_matrix(i)
        schedule_DNN(i)

    # schedule_real()
    # # 一共有k组实验
    # for num in range(10):
    #     # 9个比例循环实现
    #     for i in range(9):
    #         schedule_DNN(i)
    #         schedule_matrix(i)
