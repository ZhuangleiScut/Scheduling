import os
import pandas as pd
from pandas import DataFrame
from xlwt import Workbook
# import numpy as np
import random
import matplotlib.pyplot as plt

from src.loop.get_scheduling_DNN import schedule_DNN
from src.loop.get_scheduling_matrix import schedule_matrix
from src.loop.get_scheduling_real import schedule_real

"""
随机获取任务列表
"""
def get_task(group_num, task_num, task_total):
    # 创建文件以存储数据集id
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "data_id")
    for i in range(task_num):
        sheet1.write(i + 1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + 'group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls')

    result = pd.read_excel(path + 'group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls')
    task_list = []
    # 在task_total份图像数据中随机选取task_num份图像
    i = 0
    while i < task_num:
        temp = int(random.random() * task_total)
        if temp in task_list:
            continue
        else:
            task_list.append(temp)
            result['data_id'][i] = temp
            DataFrame(result).to_excel(path + 'group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls')
            print(i, temp)
            i += 1
    # print(ran)
    return task_list


"""
获取DNN预测时间矩阵
"""
def get_DNN_time_matrix():
    pass


"""
根据任务列表获取初始数据
"""
def get_initial_data(group_num, task_num, task_total):
    # 构造initial_data表
    # 创建文件以存储840个样本
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "image_size")
    sheet1.write(0, 2, "resolution1")
    sheet1.write(0, 3, "resolution2")
    sheet1.write(0, 4, "face_num")
    sheet1.write(0, 5, "face_area")
    sheet1.write(0, 6, "cpu_core")
    sheet1.write(0, 7, "mem_total")
    sheet1.write(0, 8, "mem_used")
    sheet1.write(0, 9, "disk_capacity")
    sheet1.write(0, 10, "frame_process_time")
    sheet1.write(0, 10, "predict_time")
    sheet1.write(0, 10, "error")
    for task in range(task_num*28):
        sheet1.write(task + 1, 0, task)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls')

    sample = pd.read_excel(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls')
    # 有28份原始数据，循环28次
    temp = 0
    for table in range(28):
        data = pd.read_excel('../../data/raw/result' + str(table + 1) + '.xlsx')

        # 循环task_num次
        for t in range(task_num):
            sample['image_size'][temp] = data['image_size'][task_list[t]]
            sample['resolution1'][temp] = data['resolution1'][task_list[t]]
            sample['resolution2'][temp] = data['resolution2'][task_list[t]]
            sample['face_num'][temp] = data['face_num'][task_list[t]]
            sample['face_area'][temp] = data['face_area'][task_list[t]]
            sample['cpu_core'][temp] = data['cpu_core'][task_list[t]]
            sample['mem_total'][temp] = data['mem_total'][task_list[t]]
            sample['mem_used'][temp] = data['mem_used'][task_list[t]]
            sample['disk_capacity'][temp] = data['disk_capacity'][task_list[t]]
            sample['frame_process_time'][temp] = data['frame_process_time'][task_list[t]]
            # 将更新写到新的Excel中
            DataFrame(sample).to_excel(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls')
            temp += 1
            print(temp)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('---new folder:', path)

if __name__ == '__main__':
    # 9个比例循环实现
    # schedule_real()
    # for i in range(9):
    #     # schedule_matrix(i)
    #     schedule_DNN(i)

    ####################################################################
    path = '../../data/loop/'
    schedule_times = 10  # 实验组数（自定义）
    task_num = 30  # 任务数（自定义）
    task_total = 300  # 任务总数
    # 一共有k组实验
    for group_num in range(schedule_times):

        # 每组实验的任务列表
        task_list = []
        # 创建根文件夹
        if not os.path.exists(path + 'group' + str(group_num)):
            mkdir(path + 'group' + str(group_num))  # 根据组号创建文件夹

        # 获取随机任务列表。先判断任务列表存不存在
        if not os.path.exists(path + 'group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls'):
            task_list = get_task(group_num, task_num, task_total)  # 输入参数为：组号，任务数，任务总数
            print('随机获取的任务列表', task_list)

        # 根据task_list任务列表获取初始数据
        # get_initial_data(task_list)
        if not os.path.exists(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls'):
            get_initial_data(group_num, task_num, task_total)  # 输入参数为：组号，任务数，任务总数


        # 9个比例循环实现
        for i in range(9):
            # 根据不同比例预测数据
            get_DNN_time_matrix(group_num, i, task_list)   # 输入组别，比例，任务列表

            schedule_DNN(group_num, task_num, i)
            schedule_matrix(group_num, task_num, i)
            schedule_real(group_num, task_num)