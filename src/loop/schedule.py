import random
import os
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import load_model
from pandas import DataFrame
from xlwt import Workbook

from src.loop.schedule_DNN import schedule_DNN
from src.loop.schedule_matrix import schedule_matrix
from src.loop.schedule_real import schedule_real

"""
实现多组实验，多比例实验
"""
"""
随机获取任务列表
"""


def get_task(group_num, task_num, task_total):
    # 创建文件以存储数据集id
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "data_id")
    for x in range(task_num):
        sheet1.write(x + 1, 0, x)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + 'group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls')
    print('---new file:', path + 'group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls')

    result = pd.read_excel(path + 'group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls')
    task_list = []
    # 在task_total份图像数据中随机选取task_num份图像
    chosen = 0
    while chosen < task_num:
        temp = int(random.random() * task_total)
        if temp in task_list:
            continue
        else:
            task_list.append(temp)
            result['data_id'][chosen] = temp
            DataFrame(result).to_excel(path + 'group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls')
            print('---choose:' + str(chosen) + 'task', str(temp))
            chosen += 1
    return task_list


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
    sheet1.write(0, 11, "predict_time")
    sheet1.write(0, 12, "error")
    for task in range(task_num * 28):
        sheet1.write(task + 1, 0, task)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls')
    print('---new file:', path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls')

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
            print('---group' + str(group_num) + ',get initial_data:', temp)


"""
新建目录
"""


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('---new folder:', path)


"""
获取DNN预测时间矩阵
"""


def get_predict_time(group_num, pro):
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
    model = load_model('../../data/DNN_train/model_0.' + str(pro + 1) + '.h5')

    # 打印模型
    model.summary()

    # 用随机数根据比例生成数据类型的选择
    real_num = int(840 * pro)
    real_data = []
    num = 0
    while num < real_num:
        real_data.append(random.randint(0, 840))
        num += 1

    # 预测
    pre = model.predict(X_train, verbose=1)
    # print(pre)
    for t in range(840):
        df['predict_time'][t] = pre[t]
        df['error'][t] = pow(df['frame_process_time'][t] - pre[t], 2)  # 计算平方误差
        # print(t, df['frame_process_time'][t], pre[t], df['error'][t])
        print('---get predict time:', t)
    DataFrame(df).to_excel(
        path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_data_0.' + str(pro + 1) + '.xls')


def get_matrix(group_num, pro):
    # 构造矩阵
    if not os.path.exists(path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_time_matrix_0.' + str(
            pro + 1) + '.xls'):
        # 构造结果表
        book = Workbook(encoding='utf-8')
        sheet1 = book.add_sheet('Sheet 1')
        sheet1.write(0, 0, "id")
        # print('task_list', task_list)
        for a in range(30):
            sheet1.write(a + 1, 0, 'task' + str(a))
            # print(task_list[a])
        for t in range(28):
            sheet1.write(0, t + 1, 'equip' + str(t))
        sheet1.write(0, 29, 'deadline')
        # 保存Excel book.save('path/文件名称.xls')
        book.save(path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_time_matrix_0.' + str(
            pro + 1) + '.xls')
        print('---new file:',
              path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_time_matrix_0.' + str(
                  pro + 1) + '.xls')

        real_num = int(84 * pro)
        real_data = []
        num = 0
        while num < real_num:
            r = random.randint(0, 840)
            if r not in real_data:
                real_data.append(r)
                print(num, r)
                num += 1

        # 打开原始数据文件
        # 打开Excel文件
        dt = pd.read_excel(
            path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_data_0.' + str(pro + 1) + '.xls')
        df = pd.read_excel(path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_time_matrix_0.' + str(
            pro + 1) + '.xls')
        for table in range(840):
            if table in real_data:
                df['equip' + str(int(table / 30))][int(table % 30)] = dt['frame_process_time'][table]
            else:
                # print('equip' + str(table % 28), int(table / 28))
                df['equip' + str(int(table / 30))][int(table % 30)] = dt['predict_time'][table]

            DataFrame(df).to_excel(
                path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_time_matrix_0.' + str(
                    pro + 1) + '.xls')
            print('---get DNN matrix:', table)
        # for table in range(840):
        #     df['equip' + str(int(table / 30))][int(table % 30)] = dt['predict_time'][table]
        # 
        #     DataFrame(df).to_excel(
        #         path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_time_matrix_0.' + str(
        #             pro + 1) + '.xls')
        #     print('---get DNN matrix:', table)


def get_predict_time_matrix(group_num, pro):  # 输入组别，比例，任务列表
    # 预测时间
    path = '../../data/loop/group' + str(group_num) + '/0.' + str(pro + 1)
    if not os.path.exists(path + '/DNN_predict_data_0.' + str(pro + 1) + '.xls'):
        get_predict_time(group_num, pro)
        # 构造矩阵
    get_matrix(group_num, pro)


"""
获取真实时间矩阵
"""


def get_real_time_matrix(group_num):
    path = '../../data/loop/'
    # 构造矩阵
    if not os.path.exists(
            path + 'group' + str(group_num) + '/real_time_matrix_' + str(group_num) + '.xls'):
        # 构造结果表
        book = Workbook(encoding='utf-8')
        sheet1 = book.add_sheet('Sheet 1')
        sheet1.write(0, 0, "id")
        for a in range(30):
            sheet1.write(a + 1, 0, 'task' + str(a))
            # print(task_list[a])
        for t in range(28):
            sheet1.write(0, t + 1, 'equip' + str(t))
        sheet1.write(0, 29, 'deadline')
        # 保存Excel book.save('path/文件名称.xls')
        book.save(path + 'group' + str(group_num) + '/real_time_matrix_' + str(group_num) + '.xls')
        print('---new file:', path + 'group' + str(group_num) + '/real_time_matrix_' + str(group_num) + '.xls')

        # 打开原始数据文件
        # 打开Excel文件
        dt = pd.read_excel(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls')
        df = pd.read_excel(
            path + 'group' + str(group_num) + '/real_time_matrix_' + str(group_num) + '.xls')
        for table in range(840):
            print('---get real_time_matrix:equip' + str(int(table / 30)), int(table % 30))
            df['equip' + str(int(table / 30))][int(table % 30)] = dt['frame_process_time'][table]

            DataFrame(df).to_excel(
                path + 'group' + str(group_num) + '/real_time_matrix_' + str(group_num) + '.xls')


if __name__ == '__main__':
    path = '../../data/loop/'
    schedule_times = 10  # 实验组数（自定义）
    task_num = 30  # 任务数（自定义）
    task_total = 300  # 任务总数
    # 一共有k组实验
    for group_num in range(10):
        print('------第' + str(group_num) + '组实验------')
        # 每组实验的任务列表
        task_list = []
        # 创建根文件夹
        if not os.path.exists(path + 'group' + str(group_num)):
            mkdir(path + 'group' + str(group_num))  # 根据组号创建文件夹

        # 获取随机任务列表。先判断任务列表存不存在
        if not os.path.exists(path + 'group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls'):
            task_list = get_task(group_num, task_num, task_total)  # 输入参数为：组号，任务数，任务总数
            print('---第' + str(group_num) + '组实验,' + '随机获取的任务列表:', task_list)

        # 根据task_list任务列表获取初始数据,方便后续的DNN预测
        if not os.path.exists(path + 'group' + str(group_num) + '/initial_data_' + str(group_num) + '.xls'):
            get_initial_data(group_num, task_num, task_total)  # 输入参数为：组号，任务数，任务总数

        # 获取真实时间矩阵,为了matrix的预测准备数据
        get_real_time_matrix(group_num)

        print('------real调度:' + str(group_num) + '组实验------')
        schedule_real(group_num, task_num)

        # 9个比例循环实现0,4,8
        for i in [0, 4, 8]:
            # 新建比例的文件夹
            if not os.path.exists(path + 'group' + str(group_num) + '/0.' + str(i + 1)):
                mkdir(path + 'group' + str(group_num) + '/0.' + str(i + 1))

            # 根据不同比例预测数据
            print('------获取DNN预测时间矩阵:' + str(group_num) + '组' + str(i) + '比例实验------')
            get_predict_time_matrix(group_num, i)  # 输入组别，比例，任务列表

            print('------DNN调度:' + str(group_num) + '组' + str(i) + '比例实验------')
            schedule_DNN(group_num, task_num, i)

            print('------matrix调度:' + str(group_num) + '组' + str(i) + '比例实验------')
            schedule_matrix(group_num, task_num, i)
        print('')
