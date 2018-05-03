import os
import random

import matplotlib.pyplot as plt
import pandas as pd
from xlwt import Workbook


def format_matrix(group, pro):
    real_num = int(48 * (pro + 1))
    real_data = []
    num = 0
    while num < real_num:
        r = random.randint(0, 480)
        if r not in real_data:
            real_data.append(r)
            print(num, r)
            num += 1

    # 打开原始数据文件
    # 打开Excel文件
    dt = pd.read_excel('./result/' + str(group) + '/0.' + str(pro + 1) + '/DNN_predict_data_0.' + str(
        pro + 1) + '.xls')
    d_DNN = pd.read_excel('./result/' + str(group) + '/0.' + str(pro + 1) + '/DNN_predict_matrix_0.' + str(
        pro + 1) + '.xls')
    d_BG = pd.read_excel('./result/' + str(group) + '/0.' + str(pro + 1) + '/BG_predict_matrix_0.' + str(
        pro + 1) + '.xls')
    d_RF = pd.read_excel('./result/' + str(group) + '/0.' + str(pro + 1) + '/RF_predict_matrix_0.' + str(
        pro + 1) + '.xls')

    # for table in range(480):
    #     df['equip' + str(int(table / 30))][int(table % 30)] = dt['predict'][table]
    #     print(dt['predict'][table])
    #     print(df['equip' + str(int(table / 30))][int(table % 30)])
    #     pd.DataFrame(df).to_excel(
    #         './result/' + str(group_num) + '/0.' + str(pro + 1) + '/DNN_predict_matrix_0.' + str(
    #             pro + 1) + '.xls')
    #     print('---get DNN matrix:', table)
    for table in range(480):
        if table in real_data:
            d_DNN['equip' + str(int(table / 30))][int(table % 30)] = dt['time'][table]
            d_BG['equip' + str(int(table / 30))][int(table % 30)] = dt['time'][table]
            d_RF['equip' + str(int(table / 30))][int(table % 30)] = dt['time'][table]

        pd.DataFrame(d_DNN).to_excel('./result/' + str(group) + '/0.' + str(pro + 1) + '/DNN_predict_matrix_0.' + str(
            pro + 1) + '.xls')
        pd.DataFrame(d_BG).to_excel(
            './result/' + str(group) + '/0.' + str(pro + 1) + '/BG_predict_matrix_0.' + str(
                pro + 1) + '.xls')
        pd.DataFrame(d_RF).to_excel(
            './result/' + str(group) + '/0.' + str(pro + 1) + '/RF_predict_matrix_0.' + str(
                pro + 1) + '.xls')
        print('---get matrix:group:' + str(group) + '-pro:' + str(pro) + '/' + str(table))


if __name__ == '__main__':
    # 组数
    group = 5
    # 比例
    pro = 9

    for g in [6,7,8,9]:

        for p in range(pro):
            format_matrix(g, p)
