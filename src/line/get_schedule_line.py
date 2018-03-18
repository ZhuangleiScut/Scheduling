import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 打开真实值调度表和预测值调度表
    data1 = pd.read_excel('../../data/scheduling_real/scheduling.xls')
    data2 = pd.read_excel('../../data/scheduling_DNN/scheduling.xls')
    data3 = pd.read_excel('../../data/scheduling_matrix/scheduling.xls')
    data4 = pd.read_excel('../../data/scheduling_duibi/scheduling_duibi.xls')

    time1 = data1['time'][0:10]
    time2 = data2['time'][0:10]
    time3 = data3['time'][0:10]
    time4 = data4['time'][0:10]

    success1 = data1['success'][0:10]
    success2 = data2['success'][0:10]
    success3 = data3['success'][0:10]
    success4 = data4['success'][0:10]

    energy1 = data1['energy'][0:10]
    energy2 = data2['energy'][0:10]
    energy3 = data3['energy'][0:10]
    energy4 = data4['energy'][0:10]

    pro1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    pro2 = data2['schedule_pro'][0:10]
    pro3 = data3['schedule_pro'][0:10]
    pro4 = data4['schedule_pro'][0:10]

    print(time1)
    print(time2)
    print(time3)
    v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # summarize history for accuracy
    plt.plot(v, time1)
    plt.plot(v, time2)
    plt.plot(v, time3)
    plt.plot(v, time4)
    # plt.plot(history.history['val_loss'])
    plt.title('time')
    plt.ylabel('time_schedule')
    plt.xlabel('num_vm')
    plt.legend(['time_real', 'time_predict', 'time_matrix','time_compared'], loc='upper left')
    plt.savefig('../../data/line/time_1.png')
    plt.show()

    # summarize history for accuracy
    plt.plot(v, success1)
    plt.plot(v, success2)
    plt.plot(v, success3)
    plt.plot(v, success4)
    # plt.plot(history.history['val_loss'])
    plt.title('offload_num')
    plt.ylabel('offload_num')
    plt.xlabel('num_vm')
    plt.legend(['offload_real', 'offload_predict', 'offload_matrix','offload_compared'], loc='upper left')
    plt.savefig('../../data/line/num_1.png')
    plt.show()

    # summarize history for accuracy
    plt.plot(v, energy1)
    plt.plot(v, energy2)
    plt.plot(v, energy3)
    plt.plot(v, energy4)
    # print('energy', energy1)
    # plt.plot(history.history['val_loss'])
    plt.title('Power consumption')
    plt.ylabel('Power consumption')
    plt.xlabel('num_vm')
    plt.legend(['real', 'predict', 'matrix','compared'], loc='upper left')
    plt.savefig('../../data/line/pow_1.png')
    plt.show()

# summarize history for accuracy
    plt.plot(v, pro1)
    plt.plot(v, pro2)
    plt.plot(v, pro3)
    plt.plot(v, pro4)
    # print('energy', energy1)
    # plt.plot(history.history['val_loss'])
    plt.title('schedule_pro')
    plt.ylabel('schedule_pro')
    plt.xlabel('num_vm')
    plt.legend(['real', 'predict', 'matrix','compared'], loc='upper left')
    plt.savefig('../../data/line/pro_1.png')
    plt.show()