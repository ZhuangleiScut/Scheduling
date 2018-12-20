import pandas as pd
from pandas import DataFrame
import random
from xlwt import Workbook


def get_random_samples(num):
    ran = []
    # 在600份图像数据中随机选取num份图像
    i = 0
    while i < num:
        temp = int(random.random() * 600)
        if temp in ran:
            continue
        else:
            ran.append(temp)
            # print(i, temp)
            i += 1
    return ran


if __name__ == '__main__':
    task_num = 600
    errors = []
    for pro in range(1, 10):
        df = pd.read_excel('./DNN_predict/sample_predict_{pro}.xls'.format(pro=pro))
        error = []
        num = task_num * pro * 0.1
        ran = get_random_samples(num)
        # print(len(ran))
        for t in range(task_num):
            print(pro,t)
            # 被抽中
            if t in ran:
                # 添加该项的误差
                error.append(df['error'][t])
            # 未被抽中
            else:
                error.append(0)

        error_avg = sum(error)/len(error)
        errors.append(error_avg)

    print(errors)
    # 作图