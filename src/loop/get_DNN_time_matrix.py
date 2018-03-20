import pandas as pd
from pandas import DataFrame
from xlwt import Workbook
# import numpy as np
import random
import matplotlib.pyplot as plt


def get_task_id(task_num):
    data = pd.read_excel('../../data/loop/task_img_id.xls')
    task_list = []
    for task in range(task_num):
        temp = data['image_id'][task]
        task_list.append(temp)
    return task_list


# 获取真实运行时间矩阵，以便后续求deadline
def get_time_matrix(pro, task_list, task_num):
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    # sheet1.write(0, 0, "id")
    for i in range(task_num):
        sheet1.write(i + 1, 0, 'task' + str(task_list[i]))
    for t in range(28):
        sheet1.write(0, t + 1, 'equip' + str(t + 1))
    sheet1.write(0, 29, 'deadline')
    # 保存Excel book.save('path/文件名称.xls')
    book.save('../../data/loop/DNN/0.'+str(pro+1)+'/predict_time_matrix0.'+str(pro+1)+'.xls')

    for i in range(28):
        # 打开原始数据文件
        # 打开Excel文件
        dt = pd.read_excel('../../data/loop/DNN/predict/0.'+str(pro+1)+'/result' + str(i + 1) + '.xlsx')
        df = pd.read_excel('../../data/loop/DNN/0.'+str(pro+1)+'/predict_time_matrix0.'+str(pro+1)+'.xls')
        for t in range(task_num):
            # 这里求deadline需要用tet真实值
            df['equip' + str(i + 1)][t] = dt['predict_time'][task_list[t]]
            DataFrame(df).to_excel('../../data/loop/DNN/0.'+str(pro+1)+'/predict_time_matrix0.'+str(pro+1)+'.xls')
        print(i)





if __name__ == '__main__':
    task_list = get_task_id(30)
    # 9种比例
    for i in range(9):
        get_time_matrix(i, task_list, 30)