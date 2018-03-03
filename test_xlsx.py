import pandas as pd
import xlwt
from xlwt import Workbook

book = Workbook(encoding='utf-8')

sheet1 = book.add_sheet('Sheet 1')

sheet1.write(0,0,"我是第一行第一列")
sheet1.write(0,1,"我是第一行第二列")

sheet1.write(1,0,"我是第2行第一列")
sheet1.write(1,1,"我是第2行第二列")

# 保存Excel book.save('path/文件名称.xls')
book.save('simple.xlsx')
data = pd.read_excel('simple.xlsx')
print(data[1][1])