import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from xlwt import Workbook
from pandas import DataFrame
# from . import mkdir
from src.loop.mkdir import mkdir
import random

"""
按比例训练模型，得到每个比例对应的最优模型
"""
"""
获取不同比例的模型，
输入参数：比例
输出参数：模型
"""

"""
创建file存储结果
"""
def mkfile(path):
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "pro")
    sheet1.write(0, 2, "error")
    # sheet1.write(0, 3, "struct")
    for i in range(100):  # 暂定每个比例测100
        sheet1.write(i + 1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path)


"""
获取model
"""
def get_model(pro):
    # 创建文件夹
    path = '../../data/DNN_train/'
    # mkdir(path+str(pro))

    data = pd.read_excel(path + str(pro) + '/models_error.xls')

    # 读取原始的数据集
    df = pd.read_excel(path + str(pro) + '/data_matrix.xls')
    # 把数据转为float类型
    # df['displacement'] = df['displacement'].astype(float)

    # 逐列获取数据集
    # First and last (mpg and car names) are ignored for X,左闭右开，13个
    X = df[df.columns[1:9]]
    print(df.columns[1:9])
    y = df['frame_process_time']

    los = 0
    # 分离数据集，将数据集按比例分为训练集和测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.5, test_size=0.5)

    # Scale the data for 收敛优化
    scaler = preprocessing.StandardScaler()

    # Set the transform parameters
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Build a 2 layer fully connected DNN with 10 and 5 units respectively
    # 初始化
    model = Sequential()
    # 添加层，参数：隐藏层的节点数，第一层的输入维数，激活函数的类型
    model.add(Dense(13, activation="relu", kernel_initializer="normal", input_dim=8))
    model.add(Dense(13, activation="relu", kernel_initializer="normal"))
    # model.add(Dense(26, activation="relu", kernel_initializer="normal"))
    # model.add(Dense(26, activation="relu", kernel_initializer="normal"))
    model.add(Dense(13, activation="relu", kernel_initializer="normal"))
    # model.add(Dense(6, activation="relu", kernel_initializer="normal"))
    model.add(Dense(1, kernel_initializer="normal"))

    # Compile the model, whith the mean squared error as a loss function
    # 编译模型，定义损失函数，均方差回归问题
    model.compile(loss='mse', optimizer='adam')
    print("编译完成。")

    # Fit the model, in 1000 epochs
    # 训练模型，1000epoch  验证集
    history = model.fit(X_train, y_train, epochs=50, validation_split=0, shuffle=True, verbose=2)
    print("训练完成。")
    # list all data in history
    print(history.history.keys())

    score = model.evaluate(X_test, y_test, verbose=1)
    # print('loss:', score[0])
    print('accuracy:', score)

    # 预测
    pre = model.predict(X_test, verbose=1)
    # print(pre)
    # print(y_test)
    erro = [abs(x - y) / x for x, y in zip(y_test, pre)]
    # print(pre)

    # # print(y_test, pre, erro)
    # print('max:', max(erro))
    # print('min:', min(erro))
    # # 归一化之后的误差
    # erro_deal = []
    # for e in erro:
    #     erro_norm = (e-min(erro))/(max(erro)-min(erro))
    #     erro_deal.append(erro_norm)
    # print('归一化误差：', sum(erro_deal)/len(erro_deal))
    los = sum(erro) / len(erro)


    temp = 0
    while os.path.exists(path + str(pro) + '/model' + str(temp) + '.h5'):
        temp += 1

    pos = int(pro / 0.1)
    print('pos', pos)
    data['pro'][temp] = pro
    data['error'][temp] = los
    DataFrame(data).to_excel(path + str(pro) + '/models_error.xls')
    print(sum(erro) / len(erro))

    # 保存模型
    model.save(path + str(pro) + '/model' + str(temp) + '.h5')


"""
用随机数的方法挑选任务
"""


def get_task(pro):  # pro表示比例的十倍
    num = pro * 84  # 数据集个数

    # 创建文件以存储数据集id
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "data_id")
    for i in range(num):
        sheet1.write(i + 1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save('../../data/DNN_train/' + str(pro) + '/data_id.xls')

    result = pd.read_excel('../../data/DNN_train/' + str(pro) + '/data_id.xls')
    ran = []
    # 在300份图像数据中随机选取30份图像
    i = 0
    while i < num:
        temp = int(random.random() * 840)
        if temp in ran:
            continue
        else:
            ran.append(temp)
            result['data_id'][i] = temp
            DataFrame(result).to_excel('../../data/DNN_train/' + str(pro) + '/data_id.xls')
            print(i, temp)
            i += 1
    # print(ran)
    return ran
"""
获取数据集矩阵
"""
def get_data_matrix(pro, ran):
    path = '../../data/DNN_train/'
    # 构造结果表
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

    for t in range(pro*84):
        sheet1.write(t+1, 0, t)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path+str(pro)+'/data_matrix.xls')

    dt = pd.read_excel("../../data/data840.xls")
    df = pd.read_excel(path + str(pro) + '/data_matrix.xls')
    for i in range(pro*84):
        # print(ran)
        df['id'][i] = ran[i]
        df['image_size'][i] = dt['image_size'][ran[i]]
        df['resolution1'][i] = dt['resolution1'][ran[i]]
        df['resolution2'][i] = dt['resolution2'][ran[i]]
        df['face_num'][i] = dt['face_num'][ran[i]]
        df['face_area'][i] = dt['face_area'][ran[i]]
        df['cpu_core'][i] = dt['cpu_core'][ran[i]]
        df['mem_total'][i] = dt['mem_total'][ran[i]]
        df['mem_used'][i] = dt['mem_used'][ran[i]]
        df['disk_capacity'][i] = dt['disk_capacity'][ran[i]]
        df['frame_process_time'][i] = dt['frame_process_time'][ran[i]]

        DataFrame(df).to_excel(path+str(pro)+'/data_matrix.xls')
        print('生成数据集矩阵:', i)


if __name__ == '__main__':
    # pro = 4  # 表示比例，用1表示0.1

    for pro in range(1, 10):
        for times in range(10):
            # 创建文件夹
            path = '../../data/DNN_train/'
            mkdir(path + str(pro))

            ran = []
            # 用随机数先选取任务数
            if not os.path.exists('../../data/DNN_train/' + str(pro) + '/data_id.xls'):
                ran = get_task(pro)
                print('任务列表', ran)

            # 构造表记录每次训练的结果
            if not os.path.exists(path + str(pro) + '/models_error.xls'):
                mkfile(path + str(pro) + '/models_error.xls')

            # 构造数据集
            if not os.path.exists(path + str(pro) + '/data_matrix.xls'):
                get_data_matrix(pro, ran)

            get_model(pro)





    # 画图像，从model_erro表中获取数据画图
    # error = pd.read_excel(path+str(pro)+'/models_error.xls')


