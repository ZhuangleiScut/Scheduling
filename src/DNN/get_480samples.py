import pandas as pd
from pandas import DataFrame
from xlwt import Workbook
# import numpy as np
import random


# 创建目标表
def build_xlsx():
    # 创建文件以存储840个样本
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
    for i in range(480):
        sheet1.write(i+1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save('data480.xls')


# 将原有的4800个样本按比例缩小到480个
def get_480samples():
    sample = pd.read_excel('data480.xls')
    # 有16份原始数据，循环28次
    temp = 0
    for i in range(16):
        data = pd.read_excel('../../data/raw2/' + str(i) + '.xls')

        # 循环300次，选取30条数据
        for t in range(300):
            if (t % 20 == 7) or (t % 20 == 15):
                sample['image_size'][temp] = data['image_size'][t]
                sample['recognition'][temp] = data['recognition'][t]
                sample['face_num'][temp] = data['face_num'][t]
                sample['face_area'][temp] = data['face_area'][t]
                sample['cpu_core'][temp] = data['cpu_core'][t]
                sample['mem'][temp] = data['mem'][t]
                sample['disk'][temp] = data['disk'][t]
                sample['time'][temp] = data['time'][t]
                # 将更新写到新的Excel中
                DataFrame(sample).to_excel('data480.xls')
                temp += 1
                print(temp)


if __name__ == '__main__':
    build_xlsx()
    get_480samples()
