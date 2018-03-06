import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 打开真实值调度表和预测值调度表
    data1 = pd.read_excel('../../data/scheduling_real/scheduling.xls')
    data2 = pd.read_excel('../../data/scheduling_DNN/scheduling.xls')
    data3 = pd.read_excel('../../data/scheduling_matrix/scheduling.xls')

    time1 = data1['time'][0:10]
    time2 = data2['time'][0:10]
    time3 = data3['time'][0:10]

    success1 = data1['success'][0:10]
    success2 = data2['success'][0:10]
    success3 = data3['success'][0:10]

    energy1 = data1['energy'][0:10]
    energy2 = data2['energy'][0:10]
    energy3 = data3['energy'][0:10]


    print(time1)
    print(time2)
    print(time3)
    v = [1,2,3,4,5,6,7,8,9,10]
    # summarize history for accuracy
    plt.plot(v, time1)
    plt.plot(v, time2)
    plt.plot(v, time3)
    # plt.plot(history.history['val_loss'])
    plt.title('time')
    plt.ylabel('time_schedule')
    plt.xlabel('num_vm')
    plt.legend(['time_real', 'time_predict', 'time_matrix'], loc='upper left')
    plt.savefig('../../data/line/time_1.png')
    plt.show()

    # summarize history for accuracy
    plt.plot(v, success1)
    plt.plot(v, success2)
    plt.plot(v, success3)
    # plt.plot(history.history['val_loss'])
    plt.title('offload_num')
    plt.ylabel('offload_num')
    plt.xlabel('num_vm')
    plt.legend(['offload_real', 'offload_predict', 'offload_matrix'], loc='upper left')
    plt.savefig('../../data/line/num_1.png')
    plt.show()

    # summarize history for accuracy
    plt.plot(v, energy1)
    plt.plot(v, energy2)
    plt.plot(v, energy3)
    # print('energy', energy1)
    # plt.plot(history.history['val_loss'])
    plt.title('Power consumption')
    plt.ylabel('Power consumption')
    plt.xlabel('num_vm')
    plt.legend(['real', 'predict', 'matrix'], loc='upper left')
    plt.savefig('../../data/line/pow_1.png')
    plt.show()
