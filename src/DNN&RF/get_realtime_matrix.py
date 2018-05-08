import pandas as pd
from xlwt import Workbook

equip_num = 16
task_num = 30


def get_task_id(path):
    data = pd.read_excel(path)
    task_list = []
    for task in range(task_num):
        temp = data['image_id'][task]
        task_list.append(temp)
    return task_list


# 获取真实运行时间矩阵，以便后续求deadline
def get_time_matrix(g, tlist):
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    # sheet1.write(0, 0, "id")
    for k in range(task_num):
        sheet1.write(k + 1, 0, 'task' + str(tlist[k]))
    for t in range(equip_num):
        sheet1.write(0, t + 1, 'equip' + str(t + 1))
    sheet1.write(0, 29, 'deadline')
    # 保存Excel book.save('path/文件名称.xls')
    book.save('time_matrix'+str(g)+'.xls')

    # 27种配置
    for e in range(equip_num):
        # 打开原始数据文件
        # 打开Excel文件
        dt = pd.read_excel('../../data/raw2/' + str(e) + '.xls')
        df = pd.read_excel('time_matrix'+str(g)+'.xls')
        for a in range(task_num):
            # 这里求deadline需要用tet真实值
            df['equip' + str(e + 1)][a] = dt['time'][tlist[a]]
            pd.DataFrame(df).to_excel('time_matrix'+str(g)+'.xls')
            print(e, a)


if __name__ == '__main__':
    for i in range(10):
        task_list = get_task_id('task_img_id'+str(i)+'.xls')
        get_time_matrix(i, task_list)
