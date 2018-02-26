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
    sheet1.write(0, 2, "resolution1")
    sheet1.write(0, 3, "resolution2")
    sheet1.write(0, 4, "face_num")
    sheet1.write(0, 5, "face_area")
    sheet1.write(0, 6, "cpu_core")
    sheet1.write(0, 7, "mem_total")
    sheet1.write(0, 8, "mem_used")
    sheet1.write(0, 9, "disk_capacity")
    sheet1.write(0, 10, "frame_process_time")
    for i in range(840):
        sheet1.write(i+1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save('../data/data840.xls')


# 将原有的8400个样本按比例缩小到840个
def get_840samples():
    sample = pd.read_excel('../data/data840.xls')
    # 有28份原始数据，循环28次
    temp = 0
    for i in range(28):
        data = pd.read_excel('../data/raw/result' + str(i + 1) + '.xlsx')

        # 循环300次，选取30条数据
        for t in range(300):
            if (t % 20 == 7) or (t % 20 == 15):
                sample['image_size'][temp] = data['image_size'][t]
                sample['resolution1'][temp] = data['resolution1'][t]
                sample['resolution2'][temp] = data['resolution2'][t]
                sample['face_num'][temp] = data['face_num'][t]
                sample['face_area'][temp] = data['face_area'][t]
                sample['cpu_core'][temp] = data['cpu_core'][t]
                sample['mem_total'][temp] = data['mem_total'][t]
                sample['mem_used'][temp] = data['mem_used'][t]
                sample['disk_capacity'][temp] = data['disk_capacity'][t]
                sample['frame_process_time'][temp] = data['frame_process_time'][t]
                # 将更新写到新的Excel中
                DataFrame(sample).to_excel('../data/data840.xls')
                temp += 1
                print(temp)


if __name__ == '__main__':
    build_xlsx()
    get_840samples()
