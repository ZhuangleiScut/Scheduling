import pandas as pd
import random

if __name__ == '__main__':
    # group_num = 0
    # pro = 0
    # time_record = pd.read_csv(
    #     '../../data/newMatrixGenerateTET/' + str(group_num) + '/matrix_predict_time_matrix_0.' + str(pro + 1) + '.csv')
    # tet = time_record['equip' + str(8)][0]
    # print(tet)
    # 用随机数根据比例生成数据类型的选择
    # real_num = int(840 * 0.1)
    # real_data = []
    # num = 0
    # while num < real_num:
    #     r = random.randint(0, 840)
    #     if r not in real_data:
    #         real_data.append(r)
    #         num += 1
    #         print(num, r)
    # for table in range(840):
    #     print('equip' + str(int(table / 30)), int(table % 30))

    real_num = int(840 * 0.1)
    real_data = []
    num = 0
    while num < real_num:
        r = random.randint(0, 840)
        if r not in real_data:
            real_data.append(r)
            num += 1
            print(num, r)
    print('real_data', real_data)


