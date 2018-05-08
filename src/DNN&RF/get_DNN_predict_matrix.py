"""
获取DNN预测时间矩阵
"""
import os
import random

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.models import load_model
from xlwt import Workbook


def get_task_id(group_num, task_num=30):
    data = pd.read_excel('task_img_id' + str(group_num) + '.xls')
    task_list = []
    for task in range(task_num):
        temp = data['image_id'][task]
        task_list.append(temp)
    return task_list


"""
根据任务列表获取初始数据
"""


def get_initial_data(group_num, task_num=30):
    # 构造initial_data表
    # 创建文件以存储480个样本

    path = './result/'

    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "image_size")
    sheet1.write(0, 2, "recognition")
    sheet1.write(0, 3, "face_num")
    sheet1.write(0, 4, "face_area")
    sheet1.write(0, 5, "cpu_core")
    sheet1.write(0, 6, "mem")
    sheet1.write(0, 7, "disk")
    sheet1.write(0, 8, "time")
    sheet1.write(0, 9, "predict")
    sheet1.write(0, 10, "error")
    for task in range(task_num * 16):
        sheet1.write(task + 1, 0, task)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + str(group_num) + '/initial_data_' + str(group_num) + '.xls')
    print('---new file:', path + str(group_num) + '/initial_data_' + str(group_num) + '.xls')

    sample = pd.read_excel(path + str(group_num) + '/initial_data_' + str(group_num) + '.xls')

    # 获取任务列表
    task_list = get_task_id(group_num)
    print('task_list', task_list)

    # 有16份原始数据，循环16次
    temp = 0
    for table in range(16):
        data = pd.read_excel('../../data/raw2/' + str(table) + '.xls')

        # 循环task_num次
        for t in range(task_num):
            sample['image_size'][temp] = data['image_size'][task_list[t]]
            sample['recognition'][temp] = data['recognition'][task_list[t]]
            sample['face_num'][temp] = data['face_num'][task_list[t]]
            sample['face_area'][temp] = data['face_area'][task_list[t]]
            sample['cpu_core'][temp] = data['cpu_core'][task_list[t]]
            sample['mem'][temp] = data['mem'][task_list[t]]
            sample['disk'][temp] = data['disk'][task_list[t]]
            sample['time'][temp] = data['time'][task_list[t]]
            # 将更新写到新的Excel中
            pd.DataFrame(sample).to_excel(path + str(group_num) + '/initial_data_' + str(group_num) + '.xls')
            temp += 1
            print('---group' + str(group_num) + ',get initial_data:', temp)


def get_predict_time(group_num, pro):
    path = './result/'
    result_index = ['image_size', 'recognition', 'face_num', 'face_area', 'cpu_core', 'mem', 'disk']
    # 读取原始的数据集
    df = pd.read_excel(path + str(group_num) + '/initial_data_' + str(group_num) + '.xls')
    # 把数据转为float类型
    # df['displacement'] = df['displacement'].astype(float)

    # 逐列获取数据集
    # First and last (mpg and car names) are ignored for X,左闭右开，13个
    X = df[result_index][0:480]
    # print(X)
    y = df['time'][0:480]

    # 分离数据集，将数据集按比例分为训练集和测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0)

    # Scale the data for 收敛优化
    scaler = preprocessing.StandardScaler()

    # Set the transform parameters
    X_train = scaler.fit_transform(X_train)

    # Build a 2 layer fully connected DNN with 10 and 5 units respectively

    # 加载模型,旧版本是load_model()
    model = load_model('./DNN_predict/0.' + str(pro + 1) + '.h5')

    # 打印模型
    model.summary()

    # 用随机数根据比例生成数据类型的选择
    # real_num = int(480 * pro)
    # real_data = []
    # num = 0
    # while num < real_num:
    #     real_data.append(random.randint(0, 4800))
    #     num += 1

    # 预测
    pre = model.predict(X_train, verbose=1)
    # print(pre)
    for t in range(480):
        df['predict'][t] = pre[t]
        df['error'][t] = pow(df['time'][t] - pre[t], 2)  # 计算平方误差
        # print(t, df['frame_process_time'][t], pre[t], df['error'][t])
        print('---get predict time:', t)
    pd.DataFrame(df).to_excel(
        './result/' + str(group_num) + '/0.' + str(pro + 1) + '/DNN_predict_data_0.' + str(pro + 1) +
        '.xls')


def get_matrix(group_num, pro):
    # 构造矩阵
    if not os.path.exists('./result/' + str(group_num) + '/0.' + str(pro + 1) + '/DNN_predict_matrix_0.' + str(
            pro + 1) + '.xls'):
        # 构造结果表
        book = Workbook(encoding='utf-8')
        sheet1 = book.add_sheet('Sheet 1')
        sheet1.write(0, 0, "id")
        # print('task_list', task_list)
        for a in range(30):
            sheet1.write(a + 1, 0, 'task' + str(a))
            # print(task_list[a])
        for t in range(16):
            sheet1.write(0, t + 1, 'equip' + str(t))
        # sheet1.write(0, 17, 'deadline')
        # 保存Excel book.save('path/文件名称.xls')
        book.save('./result/' + str(group_num) + '/0.' + str(pro + 1) + '/DNN_predict_matrix_0.' + str(
            pro + 1) + '.xls')
        print('---new file:',
              './result/' + str(group_num) + '/0.' + str(pro + 1) + '/DNN_predict_matrix_0.' + str(
                  pro + 1) + '.xls')

        # real_num = int(84 * pro)
        # real_data = []
        # num = 0
        # while num < real_num:
        #     r = random.randint(0, 840)
        #     if r not in real_data:
        #         real_data.append(r)
        #         print(num, r)
        #         num += 1
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

        # 打开原始数据文件
        # 打开Excel文件
        dt = pd.read_excel('./result/' + str(group_num) + '/0.' + str(pro + 1) + '/DNN_predict_data_0.' + str(
            pro + 1) + '.xls')
        df = pd.read_excel('./result/' + str(group_num) + '/0.' + str(pro + 1) + '/DNN_predict_matrix_0.' + str(
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
                df['equip' + str(int(table / 30))][int(table % 30)] = dt['time'][table]
            else:
                df['equip' + str(int(table / 30))][int(table % 30)] = dt['predict'][table]

            pd.DataFrame(df).to_excel(
                './result/' + str(group_num) + '/0.' + str(pro + 1) + '/DNN_predict_matrix_0.' + str(
                    pro + 1) + '.xls')
            print('---get DNN matrix:', table)


def get_predict_time_matrix(group_num, pro):  # 输入组别，比例，任务列表
    # 预测时间
    path = './result/' + str(group_num) + '/0.' + str(pro + 1) + '/'
    if not os.path.exists(path + 'DNN_predict_data_0.' + str(pro + 1) + '.xls'):
        get_predict_time(group_num, pro)
        # 构造矩阵
    get_matrix(group_num, pro)


if __name__ == '__main__':
    # 组数
    group = 10
    # 9种比例
    pro = 9

    path = './result/'

    # 10组
    for g in range(group):

        # 根据task_list任务列表获取初始数据,方便后续的DNN预测
        if not os.path.exists(path + str(g) + '/initial_data_' + str(g) + '.xls'):
            get_initial_data(g)  # 输入参数为：组号，任务数，任务总数

        for p in range(pro):
            print(' - 获取DNN预测时间矩阵:' + str(g) + '组' + str(p) + '比例')
            get_predict_time_matrix(g, p)  # 输入组别，比例，任务列表
