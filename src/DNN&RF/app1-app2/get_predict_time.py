import os
import pandas as pd
import time
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from xlwt import Workbook
from pandas import DataFrame
from src.loop.mkdir import mkdir
import random
from keras.models import load_model


# pro从1到9
def get_model(pro):
    # 创建文件夹
    path = './DNN_predict/'
    # mkdir(path+str(pro))
    result_index = ['cpu', 'mem', 'ping']
    input_len = len(result_index)

    data = pd.read_excel(path + str(pro) + '/models_error.xls')

    # 读取原始的数据集
    df = pd.read_excel('./DNN_predict/sample_predict.xls')
    # 把数据转为float类型
    # df['displacement'] = df['displacement'].astype(float)

    # 逐列获取数据集
    # First and last (mpg and car names) are ignored for X,左闭右开，13个
    X = df[result_index]
    # print(df.columns[1:9])
    y = df['tet']

    los = 0
    # 分离数据集，将数据集按比例分为训练集和测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0)

    # Scale the data for 收敛优化
    scaler = preprocessing.StandardScaler()

    # Set the transform parameters
    X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # 加载模型,旧版本是load_model()
    model = load_model('./DNN_predict/model_{p}.h5'.format(p=pro))

    # 打印模型
    model.summary()

    # 预测
    pre = model.predict(X_train, verbose=1)
    print(pre)
    for t in range(len(y)):
        df['tet_predict'][t] = pre[t]
        df['error'][t] = pow((df['tet'][t] - pre[t])/df['tet'][t], 2)  # 计算平方误差
        print(t, df['tet'][t], pre[t], df['error'][t])
    DataFrame(df).to_excel('./DNN_predict/sample_predict_{pro}.xls'.format(pro=pro))


if __name__ == '__main__':
    for pro in range(1, 10):
        get_model(pro)

