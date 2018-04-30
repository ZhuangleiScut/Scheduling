
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
        # for table in range(840):
        #     if table in real_data:
        #         df['equip' + str(int(table / 30))][int(table % 30)] = dt['frame_process_time'][table]
        #     else:
        #         # print('equip' + str(table % 28), int(table / 28))
        #         df['equip' + str(int(table / 30))][int(table % 30)] = dt['predict_time'][table]
        #
        #     DataFrame(df).to_excel(
        #         path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_time_matrix_0.' + str(
        #             pro + 1) + '.xls')
        #     print('---get DNN matrix:', table)
        for table in range(840):
            df['equip' + str(int(table / 30))][int(table % 30)] = dt['predict_time'][table]

            DataFrame(df).to_excel(
                path + 'group' + str(group_num) + '/0.' + str(i + 1) + '/DNN_predict_time_matrix_0.' + str(
                    pro + 1) + '.xls')
            print('---get DNN matrix:', table)

def get_predict_time_matrix(group_num, pro):  # 输入组别，比例，任务列表
    # 预测时间
    path = '../../data/loop/group' + str(group_num) + '/0.' + str(pro + 1)
    if not os.path.exists(path + '/DNN_predict_data_0.' + str(pro + 1) + '.xls'):
        get_predict_time(group_num, pro)
        # 构造矩阵
    get_matrix(group_num, pro)


if __name__ == '__main__':
    # 组数
    group = 10
    # 9种比例
    pro = 9

    # 10组
    for g in range(group):
        for p in range(pro):
            print(' - 获取DNN预测时间矩阵:' + str(g) + '组' + str(p) + '比例')
            get_predict_time_matrix(g, i)  # 输入组别，比例，任务列表