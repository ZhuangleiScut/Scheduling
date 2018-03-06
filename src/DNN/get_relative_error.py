import pandas as pd
from pandas import DataFrame
from xlwt import Workbook
# import numpy as np
import random
import matplotlib.pyplot as plt


def get_error1():
    for pro in range(1):
        # result = 0
        sum = 0

        # 每个配置的表
        for equip in range(28):
            data = pd.read_excel('../../data/scheduling_DNN/predict/0.'+str(pro + 1)+'/result' + str(equip + 1) + '.xlsx')

            # 每个表300份数据
            for d in range(300):
                error = data['error'][d]
                real_time = data['frame_process_time'][d]
                err = error / pow(real_time, 2)
                print('error', err)
                sum += err
                print('sum', sum)

        # 对于每个比例的误差进行记录
        result = pd.read_excel('../../data/scheduling_DNN/relative_error.xls')
        result['error1'][pro] = sum
        result['pro'][pro] = pro + 1
        DataFrame(result).to_excel('../../data/scheduling_DNN/relative_error.xls')
    # return sum


def get_error2():
    for pro in range(1):
        # result = 0
        sum_error = 0
        sum_real =0
        # 每个配置的表
        for equip in range(28):
            data = pd.read_excel('../../data/scheduling_DNN/predict/0.'+str(pro + 1)+'/result' + str(equip + 1) + '.xlsx')

            # 每个表300份数据
            for d in range(300):
                error = data['error'][d]
                real_time = data['frame_process_time'][d]

                sum_error += error
                sum_real += pow(real_time, 2)
                # err = error / pow(real_time, 2)
                # print('error', err)
                # sum += err
                # print('sum', sum)

        # 对于每个比例的误差进行记录
        result = pd.read_excel('../../data/scheduling_DNN/relative_error.xls')
        result['error1'][pro] = sum_error / sum_real
        DataFrame(result).to_excel('../../data/scheduling_DNN/relative_error.xls')


def get_ralative_error():
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "pro")
    sheet1.write(0, 2, "error1")
    sheet1.write(0, 3, "error2")
    for t in range(10):
        sheet1.write(t + 1, 0, t)
    # 保存Excel book.save('path/文件名称.xls')
    book.save('../../data/scheduling_DNN/relative_error.xls')

    #############################################################
    get_error1()
    # get_error2()

if __name__ == '__main__':
    get_ralative_error()
