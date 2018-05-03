import pandas as pd
from pandas import DataFrame
import random
from xlwt import Workbook


# 创建目标表
def build_xlsx(num):
    # 创建文件以存储480个样本
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "image_id")
    for i in range(30):
        sheet1.write(i + 1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save('task_img_id'+str(num)+'.xls')


# 挑选30张图像进行调度，用随机函数从300张图像中挑选30张
def get_random_samples(num, g):
    result = pd.read_excel('task_img_id'+str(g)+'.xls')
    ran = []
    # 在300份图像数据中随机选取30份图像
    p = 0
    while p < 30:
        temp = int(random.random() * 300)
        if temp in ran:
            continue
        else:
            ran.append(temp)
            result['image_id'][p] = temp
            DataFrame(result).to_excel('task_img_id'+str(g)+'.xls')
            print(p, temp)
            p += 1


if __name__ == '__main__':
    num = 30
    for i in range(10):
        build_xlsx(i)
        get_random_samples(num, i)
