import os
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import pandas as pd
from xlwt import Workbook


def get_RF_predict(group, pro):
    path = './result/' + str(group) + '/'
    # 读取原始的数据集
    df = pd.read_excel(path + 'initial_data_' + str(group) + '.xls')
    # 逐列获取数据集
    # First and last (mpg and car names) are ignored for X,左闭右开，13个
    x = df[df.columns[1:8]]
    y = df['time']
    # print(x)
    # print(y)

    # 划分训练集和测试集x表示样本特征集，y表示样本结果  test_size 样本占比,random_state 随机数的种子
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.1 * (pro + 1), random_state=1)

    # 划分训练集和测试集x表示样本特征集，y表示样本结果  test_size 样本占比,random_state 随机数的种子
    p_train, p_test, q_train, q_test = train_test_split(x, y, test_size=0, random_state=1)

    # Scale the data for 收敛优化
    scaler = preprocessing.StandardScaler()
    # Set the transform parameters
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # 随机森林 n_estimators：决策树的个数,越多越好,不过值越大，性能就会越差,至少100
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(x_train, y_train)
    p_rf = rf.predict(p_train)

    # Bagging估计
    bg = BaggingRegressor(DecisionTreeRegressor())
    bg.fit(x_train, y_train)
    p_bg = bg.predict(p_train)

    # print('lr', y_lr)
    print('rf', p_rf)
    print('bg', p_bg)
    print(p_train)
    print(len(p_rf))

    # 构造矩阵
    if not os.path.exists('./result/' + str(group) + '/0.' + str(pro + 1) + '/RF_predict_matrix_0.' + str(
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
        book.save('./result/' + str(group) + '/0.' + str(pro + 1) + '/RF_predict_matrix_0.' + str(
            pro + 1) + '.xls')
        print('---new file:',
              './result/' + str(group) + '/0.' + str(pro + 1) + '/BG_predict_matrix_0.' + str(
                  pro + 1) + '.xls')

    # 构造矩阵
    if not os.path.exists('./result/' + str(group) + '/0.' + str(pro + 1) + '/BG_predict_matrix_0.' + str(
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
        book.save('./result/' + str(group) + '/0.' + str(pro + 1) + '/BG_predict_matrix_0.' + str(
            pro + 1) + '.xls')
        print('---new file:',
              './result/' + str(group) + '/0.' + str(pro + 1) + '/BG_predict_matrix_0.' + str(
                  pro + 1) + '.xls')

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
    dRF = pd.read_excel('./result/' + str(group) + '/0.' + str(pro + 1) + '/RF_predict_matrix_0.' + str(
        pro + 1) + '.xls')
    dBG = pd.read_excel('./result/' + str(group) + '/0.' + str(pro + 1) + '/BG_predict_matrix_0.' + str(
        pro + 1) + '.xls')

    # for table in range(480):
    #     dRF['equip' + str(int(table / 30))][int(table % 30)] = p_rf[table]
    #     dBG['equip' + str(int(table / 30))][int(table % 30)] = p_bg[table]
    #     # print(dt['predict'][table])
    #     # print(df['equip' + str(int(table / 30))][int(table % 30)])
    #     pd.DataFrame(dRF).to_excel(
    #         './result/' + str(group) + '/0.' + str(pro + 1) + '/RF_predict_matrix_0.' + str(
    #             pro + 1) + '.xls')
    #     pd.DataFrame(dBG).to_excel(
    #         './result/' + str(group) + '/0.' + str(pro + 1) + '/BG_predict_matrix_0.' + str(
    #             pro + 1) + '.xls')
    #     print('---get matrix:', table)

    for table in range(480):
        if table in real_data:
            dRF['equip' + str(int(table / 30))][int(table % 30)] = dt['time'][table]
            dBG['equip' + str(int(table / 30))][int(table % 30)] = dt['time'][table]
        else:
            dRF['equip' + str(int(table / 30))][int(table % 30)] = p_rf[table]
            dBG['equip' + str(int(table / 30))][int(table % 30)] = p_bg[table]

        pd.DataFrame(dRF).to_excel(
            './result/' + str(group) + '/0.' + str(pro + 1) + '/RF_predict_matrix_0.' + str(
                pro + 1) + '.xls')
        pd.DataFrame(dBG).to_excel(
            './result/' + str(group) + '/0.' + str(pro + 1) + '/BG_predict_matrix_0.' + str(
                pro + 1) + '.xls')
        print('---get DNN matrix:', table)


if __name__ == "__main__":
    # 10组
    # 9种比例

    for g in [9]:
        for p in range(9):
            get_RF_predict(g, p)
