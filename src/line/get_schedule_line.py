import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 打开真实值调度表和预测值调度表
    data1 = pd.read_excel('../../data/scheduling_real/scheduling.xls')
    data2 = pd.read_excel('../../data/scheduling_DNN/scheduling.xls')

    time1 = data1['time'][0:10]
    time2 = data2['time'][0:10]

    success1 = data1['success'][0:10]
    success2 = data2['success'][0:10]
    print(time1)
    print(time2)
    v = [1,2,3,4,5,6,7,8,9,10]
    # summarize history for accuracy
    plt.plot(v, time1)
    plt.plot(v, time2)
    # plt.plot(history.history['val_loss'])
    plt.title('time')
    plt.ylabel('time_schedule')
    plt.xlabel('num_vm')
    plt.legend(['time_real','time_predict'], loc='upper right')
    plt.savefig('../../data/line/time_1.png')
    plt.show()

    # summarize history for accuracy
    plt.plot(v, success1)
    plt.plot(v, success2)
    # plt.plot(history.history['val_loss'])
    plt.title('offload_num')
    plt.ylabel('offload_num')
    plt.xlabel('num_vm')
    plt.legend(['offload_real', 'offload_predict'], loc='upper right')
    plt.savefig('../../data/line/num_1.png')
    plt.show()


    power1 = []
    power2 = []
    for g in range(10):
        power1.append(1.75*time1[g])
        power2.append(1.75*time2[g])

    # summarize history for accuracy
    plt.plot(v, power1)
    plt.plot(v, power2)
    # plt.plot(history.history['val_loss'])
    plt.title('Power consumption')
    plt.ylabel('Power consumption')
    plt.xlabel('num_vm')
    plt.legend(['real', 'predict'], loc='upper right')
    plt.savefig('../../data/line/pow_1.png')
    plt.show()
