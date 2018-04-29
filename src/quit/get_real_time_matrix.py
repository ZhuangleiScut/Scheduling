import os
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import load_model
from pandas import DataFrame
from xlwt import Workbook


def get_matrix(group_num, pro):
    # 构造矩阵
    if not os.path.exists(
            path + 'group' + str(group_num) + '/0.' + str(0 + 1) + '/real_time_matrix_' + str(pro) + '.xls'):
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
        book.save(path + 'group' + str(group_num) + '/0.' + str(0 + 1) + '/real_time_matrix_' + str(pro) + '.xls')

        # 打开原始数据文件
        # 打开Excel文件
        dt = pd.read_excel(path + 'group' + str(group_num) + '/initial_data_' + str(pro) + '.xls')
        df = pd.read_excel(
            path + 'group' + str(group_num) + '/0.' + str(0 + 1) + '/real_time_matrix_' + str(pro) + '.xls')
        for table in range(840):
            print('equip' + str(table % 28), int(table / 28))
            df['equip' + str(table % 28)][int(table / 28)] = dt['frame_process_time'][table]

            DataFrame(df).to_excel(
                path + 'group' + str(group_num) + '/0.' + str(0 + 1) + '/real_time_matrix_' + str(pro) + '.xls')
        # for k in range(28):
        #
        #     for t in range(task_num):
        #         # 这里求deadline需要用tet真实值
        #         df['equip' + str(k + 1)][t] = dt['predict_time'][task_list[t]]
        #         DataFrame(df).to_excel('../../data/scheduling_DNN/predict_time_matrix0.' + str(pro + 1) + '.xls')
        #     print(k)

    # 读取原始的数据集


if __name__ == '__main__':
    path = '../../data/loop/'
    get_matrix(0, 0)
