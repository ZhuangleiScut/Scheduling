import random
import os
import pandas as pd
from pandas import DataFrame
from xlwt import Workbook


def get_energy(group, type, pro=None):
    path = '../../data/random_loop/group' + str(group) + '/'
    if type == 'real':
        result = pd.read_excel(path + type + '/scheduling_real.xls')

        # 10种虚拟机配置
        for i in range(1, 11):
            data = pd.read_excel(path + type + '/schedule/' + str(i) + '_real.xlsx')
            print(' - group:', group)
            print(' - vm:', i)

            energy_sum = 0
            # 任务数
            for t in range(30):
                offload = data['offload'][t]
                # print('offload', offload)
                if offload == 2 or offload == 0:
                    # 树莓派的功率10W，乘上deadline
                    energy_sum += 10 * data['deadline'][t]
                    # print(data['task'][t])
            result['energy'][i - 1] = energy_sum
            # result['energy'][i] = None
            print(energy_sum)
            DataFrame(result).to_excel(path + type + '/scheduling_real.xls')
    if type == 'DNN' or type == 'matrix':
        result = pd.read_excel(path + str(pro) + '/' + type + '/scheduling_' + type + '.xls')

        # 10种虚拟机配置
        for i in range(1, 11):
            if os.path.exists(path + str(pro) + '/' + type + '/schedule/' + str(i) + '_' + type + '.xlsx'):
                data = pd.read_excel(path + str(pro) + '/' + type + '/schedule/' + str(i) + '_' + type + '.xlsx')
                print(' - group:', group)
                print(' - vm:', i)

                energy_sum = 0
                # 任务数
                for t in range(30):
                    offload = data['offload'][t]
                    # print('offload', offload)
                    if offload == 2 or offload == 0:
                        # 树莓派的功率10W，乘上deadline
                        energy_sum += 10 * data['deadline'][t]
                        # print(data['task'][t])
                result['energy'][i - 1] = energy_sum
                # result['energy'][i] = None
                print(energy_sum)
                DataFrame(result).to_excel(path + str(pro) + '/' + type + '/scheduling_' + type + '.xls')

            else:
                data = pd.read_excel(path + str(pro) + '/' + type + '/schedule/' + str(i) + '.xlsx')
                print(' - group:', group)
                print(' - vm:', i)

                energy_sum = 0
                # 任务数
                for t in range(30):
                    offload = data['offload'][t]
                    # print('offload', offload)
                    if offload == 2 or offload == 0:
                        # 树莓派的功率10W，乘上deadline
                        energy_sum += 10 * data['deadline'][t]
                        # print(data['task'][t])
                result['energy'][i - 1] = energy_sum
                # result['energy'][i] = None
                print(energy_sum)
                DataFrame(result).to_excel(path + str(pro) + '/' + type + '/scheduling_' + type + '.xls')



if __name__ == '__main__':
    # 组数
    group = 10
    # 比例
    pro = 3

    for g in range(group):
        # real
        get_energy(g, 'real')

        for p in [0.1, 0.5, 0.9]:
            get_energy(g, 'DNN', p)
            get_energy(g, 'matrix', p)
