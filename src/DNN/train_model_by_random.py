import matplotlib.pyplot as plt
import os
import pandas as pd
import time
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
    sheet1.write(0, 3, "DNN_1")
    sheet1.write(0, 4, "DNN_2")
    sheet1.write(0, 5, "DNN_3")
    sheet1.write(0, 6, "DNN_4")
    sheet1.write(0, 7, "DNN_5")
    sheet1.write(0, 8, "DNN_6")
    sheet1.write(0, 9, "DNN_7")
    sheet1.write(0, 10, "DNN_8")
    sheet1.write(0, 11, "DNN_9")
    sheet1.write(0, 12, "DNN_10")

    # sheet1.write(0, 3, "struct")
    for i in range(100):  # 暂定每个比例测100
        sheet1.write(i + 1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path)


"""
获取model
"""


# pro从1到9
def get_model(pro):
    # 创建文件夹
    path = './DNN_predict/'
    # mkdir(path+str(pro))
    result_index = ['image_size', 'recognition', 'face_num', 'face_area', 'cpu_core', 'mem', 'disk']
    input_len = len(result_index)

    data = pd.read_excel(path + str(pro) + '/models_error.xls')

    # 读取原始的数据集
    df = pd.read_csv('samples.csv')
    # 把数据转为float类型
    # df['displacement'] = df['displacement'].astype(float)

    # 逐列获取数据集
    # First and last (mpg and car names) are ignored for X,左闭右开，13个
    X = df[result_index]
    # print(df.columns[1:9])
    y = df['time']

    los = 0
    # 分离数据集，将数据集按比例分为训练集和测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=pro * 0.1)

    # Scale the data for 收敛优化
    scaler = preprocessing.StandardScaler()

    # Set the transform parameters
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Build a 2 layer fully connected DNN with 10 and 5 units respectively
    # 初始化
    model = Sequential()

    # 随机构造神经网络模型,隐藏层数
    ran_lay = random.randint(2, 10)
    DNN_struct = []

    # 添加层，参数：隐藏层的节点数，第一层的输入维数，激活函数的类型
    for lay in range(ran_lay):  # 对于每一层
        if lay == 0:
            r = random.randint(5, 15)
            DNN_struct.append(r)
            model.add(Dense(r, activation="relu", kernel_initializer="normal", input_dim=input_len))
        else:
            r = random.randint(5, 15)
            DNN_struct.append(r)
            model.add(Dense(r, activation="relu", kernel_initializer="normal"))

    # 最后一层
    model.add(Dense(1, kernel_initializer="normal"))

    print("神经网络结构：", DNN_struct)
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
    los = sum(erro) / len(erro)

    temp = 0
    while os.path.exists(path + str(pro) + '/model' + str(temp) + '.h5'):
        temp += 1

    pos = int(pro / 0.1)
    print('pos', pos)
    data['pro'][temp] = pro
    data['error'][temp] = los
    for length in range(len(DNN_struct)):
        data['DNN_' + str(length + 1)][temp] = DNN_struct[length]
    DataFrame(data).to_excel(path + str(pro) + '/models_error.xls')
    print(sum(erro) / len(erro))

    # 保存模型
    model.save(path + str(pro) + '/model' + str(temp) + '.h5')


"""
用随机数的方法挑选任务
"""


def get_task(pro):  # pro表示比例的十倍
    num = pro * 480  # 数据集个数

    # 创建文件以存储数据集id
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "data_id")
    for i in range(num):
        sheet1.write(i + 1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save('./DNN_predict/' + str(pro) + '/data_id.xls')

    result = pd.read_excel('./DNN_predict/' + str(pro) + '/data_id.xls')
    ran = []
    # 在300份图像数据中随机选取30份图像
    i = 0
    while i < num:
        temp = int(random.random() * 4800)
        if temp in ran:
            continue
        else:
            ran.append(temp)
            result['data_id'][i] = temp
            DataFrame(result).to_excel('./DNN_predict/' + str(pro) + '/data_id.xls')
            print(i, temp)
            i += 1
    # print(ran)
    return ran


"""
获取数据集矩阵
"""


def get_data_matrix(pro, ran):
    path = './DNN_predict/'
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "image_size")
    sheet1.write(0, 2, "resolution")
    sheet1.write(0, 3, "face_num")
    sheet1.write(0, 4, "face_area")
    sheet1.write(0, 5, "cpu_core")
    sheet1.write(0, 6, "mem")
    sheet1.write(0, 7, "disk")
    sheet1.write(0, 8, "time")

    for t in range(pro * 480):
        sheet1.write(t + 1, 0, t)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + str(pro) + '/data_matrix.xls')

    # dt = pd.read_excel("../../data/data840.xls")
    df = pd.read_excel(path + str(pro) + '/data_matrix.xls')
    for i in range(pro * 480):
        # dt = pd.read_excel('../../data/raw2/'+data840.xls")
        # print(ran)
        task = ran[i]
        dt = pd.read_excel('../../data/raw2/' + str(int(task / 300)) + '.xls')
        df['id'][i] = task
        df['image_size'][i] = dt['image_size'][ran[i] % 300]
        df['resolution'][i] = dt['recognition'][ran[i] % 300]
        df['face_num'][i] = dt['face_num'][ran[i] % 300]
        df['face_area'][i] = dt['face_area'][ran[i] % 300]
        df['cpu_core'][i] = dt['cpu_core'][ran[i] % 300]
        df['mem'][i] = dt['mem'][ran[i] % 300]
        df['disk'][i] = dt['disk'][ran[i] % 300]
        df['time'][i] = dt['time'][ran[i] % 300]
        print('生成数据集矩阵:', i)
    DataFrame(df).to_excel(path + str(pro) + '/data_matrix.xls')


if __name__ == '__main__':
    # pro = 4  # 表示比例，用1表示0.1
    # pro = 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

    """
    建个表保存时间
    """
    path0 = './DNN_predict/'
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "pro")
    sheet1.write(0, 2, "time")
    for t in range(10):
        sheet1.write(t + 1, 0, t)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path0 + 'train_time.xls')

    """
    九种比例
    """
    for pro in [7, 8, 9]:
        """
        每种比例50次
        """
        data = pd.read_excel(path0 + '/train_time.xls')
        time1 = time.time()
        print('---start:', time1)
        for times in range(50):
            # 创建文件夹
            path = './DNN_predict/'
            mkdir(path + str(pro))

            ran = []
            # 用随机数先选取任务数
            # if not os.path.exists(path + str(pro) + '/data_id.xls'):
            #     ran = get_task(pro)
            #     print('任务列表', ran)

            # 构造表记录每次训练的结果
            if not os.path.exists(path + str(pro) + '/models_error.xls'):
                mkfile(path + str(pro) + '/models_error.xls')

            # 构造数据集
            # if not os.path.exists(path + str(pro) + '/data_matrix.xls'):
            #     get_data_matrix(pro, ran)

            print('---开始训练---')
            get_model(pro)
        time2 = time.time()
        print('---end:', time2)
        time_avg = (time2 - time1) / 50
        data['pro'][pro - 1] = pro
        data['time'][pro - 1] = time_avg
        DataFrame(data).to_excel(path0 + '/train_time.xls')
