import os

import pandas as pd
from pandas import DataFrame
from xlwt import Workbook

i = 0;
while os.path.exists(str(i) + '.xls'):
    i += 1
if not os.path.exists(str(i) + '.xls'):
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    for a in range(300):  # y
        sheet1.write(a + 1, 0, a)
        # print(task_list[a])
    sheet1.write(0, 1, 'task')
    sheet1.write(0, 2, 'time')
    # 保存Excel book.save('path/文件名称.xls')
    book.save(str(i) + '.xls')
    print('建表完成。')