import os

import pandas as pd
from pandas import DataFrame
from xlwt import Workbook

if not os.path.exists('0.xls'):
    if not os.path.exists('0.xls'):
        # 构造结果表
        book = Workbook(encoding='utf-8')
        sheet1 = book.add_sheet('Sheet 1')
        sheet1.write(0, 0, "id")
        for a in range(30):  # 纵轴
            sheet1.write(a + 1, 0, 'task' + str(a))
            # print(task_list[a])
        for t in range(28):  # 横轴
            sheet1.write(0, t + 1, 'equip' + str(t))
        sheet1.write(0, 29, 'deadline')
        # 保存Excel book.save('path/文件名称.xls')
        book.save('0.xls')
        print('建表完成。')


    # 打开原始数据文件
    # 打开Excel文件
    dt = pd.read_excel('0.xls')
    df = pd.read_excel('0.xls')
    for table in range(840):
        print('---get real_time_matrix:equip' + str(int(table / 30)), int(table % 30))
        df['equip' + str(int(table / 30))][int(table % 30)] = dt['frame_process_time'][table]

        DataFrame(df).to_excel('0.xls')
