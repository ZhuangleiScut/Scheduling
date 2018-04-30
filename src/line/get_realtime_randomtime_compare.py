import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 打开真实值调度表和预测值调度表
    # 序号
    group = 9
    propro = 0.5
    path1 = '../../data/loop/'
    path2 = '../../data/random_loop/'

    # 序号
    for i in range(10):
        data1 = pd.read_excel(path1 + 'group' + str(i) + '/real/scheduling_real.xls')
        data2 = pd.read_excel(path2 + 'group' + str(i) + '/real/scheduling_real.xls')

        time1 = data1['time'][0:10]
        time2 = data2['time'][0:10]

        success1 = data1['success'][0:10]
        success2 = data2['success'][0:10]

        energy1 = data1['energy'][0:10]
        energy2 = data2['energy'][0:10]

        print(time1)
        print(time2)

        v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # summarize history for accuracy
        plt.plot(v, time1)
        plt.plot(v, time2)

        plt.title('time-'+str(i))
        plt.ylabel('time_schedule')
        plt.xlabel('num_vm')
        plt.legend(['time_real', 'time_radom'], loc='lower left')
        plt.savefig('../../data/line/duibi/time_'+str(i)+'.png')
        plt.show()


    # summarize history for accuracy
    # plt.plot(v, success1)
    # plt.plot(v, success2)
    # plt.title('offload_num')
    # plt.ylabel('offload_num')
    # plt.xlabel('num_vm')
    # plt.legend(['offload_real', 'offload_random'], loc='upper left')
    # # plt.savefig('../../data/line/num_1.png')
    # plt.show()
