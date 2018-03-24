import os
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import load_model
from pandas import DataFrame


"""
通过加载get_model得到的模型
对数据集进行预测
并将得到的结果矩阵保存
"""
path = '../../data/loop/'
def get_predict_time(group_num, i, task_list):
    result_index = ['image_size', 'resolution1', 'resolution2', 'face_num', 'face_area', 'cpu_core', 'mem_total',
                    'mem_used']
    # 读取原始的数据集
    df = pd.read_excel(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls')
    # 把数据转为float类型
    # df['displacement'] = df['displacement'].astype(float)

    # 逐列获取数据集
    # First and last (mpg and car names) are ignored for X,左闭右开，13个
    X = df[result_index][0:840]
    # print(X)
    y = df['frame_process_time'][0:840]

    # 分离数据集，将数据集按比例分为训练集和测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0)

    # Scale the data for 收敛优化
    scaler = preprocessing.StandardScaler()

    # Set the transform parameters
    X_train = scaler.fit_transform(X_train)

    # Build a 2 layer fully connected DNN with 10 and 5 units respectively

    # 加载模型,旧版本是load_model()
    model = load_model('../../data/DNN_train/model_0.'+str(i+1)+'.h5')

    # 打印模型
    model.summary()

    # 预测
    pre = model.predict(X_train, verbose=1)
    print(pre)
    for t in range(840):
        df['predict_time'][t] = pre[t]
        df['error'][t] = pow(df['frame_process_time'][t] - pre[t], 2)  # 计算平方误差
        print(t, df['frame_process_time'][t], pre[t], df['error'][t])
    DataFrame(df).to_excel(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls')


def get_matrix(group_num, i, task_list):
    # 构造矩阵
    if not os.path.exists(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls'):
        get_initial_data(group_num, task_num, task_total)  # 输入参数为：组号，任务数，任务总数
    # 读取原始的数据集
    df = pd.read_excel(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls')


def get_predict_time_matrix(group_num, i, task_list):  # 输入组别，比例，任务列表
    # 预测时间
    get_predict_time(group_num, i, task_list)
    # 构造矩阵
    get_matrix(group_num, i, task_list)