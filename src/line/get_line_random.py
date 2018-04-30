import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    path = '../../data/'
    DNN = []
    matrix = []
    real = []
    DNN_random = []
    matrix_random = []
    real_random = []

    delay = []
    offload = []
    pro = 0.5
    # 个数
    group = 10

    # 读取表,10组
    for i in range(group):
        real_i = pd.read_excel(path + "loop/group" + str(i) + "/real/scheduling_real.xls")
        DNN_i = pd.read_excel(path + "loop/group" + str(i) + '/' + str(pro) + "/DNN/scheduling_DNN.xls")
        martix_i = pd.read_excel(path + "loop/group" + str(i) + '/' + str(pro) + "/matrix/scheduling_matrix.xls")

        real_random_i = pd.read_excel(path + "random_loop/group" + str(i) + "/real/scheduling_real.xls")
        DNN_random_i = pd.read_excel(path + "random_loop/group" + str(i) + '/' + str(pro) + "/DNN/scheduling_DNN.xls")
        martix_random_i = pd.read_excel(path + "random_loop/group" + str(i) + '/' + str(pro) + "/matrix"
                                                                                               "/scheduling_matrix.xls")

        DNN.append(DNN_i)
        matrix.append(martix_i)
        real.append(real_i)
        DNN_random.append(DNN_random_i)
        matrix_random.append(martix_random_i)
        real_random.append(real_random_i)

    time1 = []
    time2 = []
    time3 = []
    time4 = []
    time5 = []
    time6 = []

    num1 = []
    num2 = []
    num3 = []
    num4 = []
    num5 = []
    num6 = []
    # 10种虚拟机配置
    for k in range(10):
        tr = []
        nr = []
        td = []
        nd = []
        tm = []
        nm = []

        trr = []
        nrr = []
        tdr = []
        ndr = []
        tmr = []
        nmr = []
        for t in range(group):
            # 算10组平均值
            r = real[t]
            d = DNN[t]
            m = matrix[t]
            rr = real_random[t]
            dr = DNN_random[t]
            mr = matrix_random[t]

            tr.append(r['time'][k])
            nr.append(r['success'][k])

            td.append(d['time'][k])
            nd.append(d['success'][k])

            tm.append(m['time'][k])
            nm.append(m['success'][k])

            trr.append(rr['time'][k])
            nrr.append(rr['success'][k])

            tdr.append(dr['time'][k])
            ndr.append(dr['success'][k])

            tmr.append(mr['time'][k])
            nmr.append(mr['success'][k])
            # print(tr,nr)
        tr_avg = sum(tr)/len(tr)
        nr_avg = sum(nr)/len(nr)
        td_avg = sum(td)/len(td)
        nd_avg = sum(nd)/len(nd)
        tm_avg = sum(tm)/len(tm)
        nm_avg = sum(nm)/len(nm)

        trr_avg = sum(trr)/len(trr)
        nrr_avg = sum(nrr)/len(nrr)
        tdr_avg = sum(tdr)/len(tdr)
        ndr_avg = sum(ndr)/len(ndr)
        tmr_avg = sum(tmr)/len(tmr)
        nmr_avg = sum(nmr)/len(nmr)

        # print(nmr, nmr_avg)
        # print(ndr, ndr_avg)
        # print(tr, tr_avg)
        time1.append(tr_avg)
        time2.append(td_avg)
        time3.append(tm_avg)
        time4.append(trr_avg)
        time5.append(tdr_avg)
        time6.append(tmr_avg)

        num1.append(nr_avg)
        num2.append(nd_avg)
        num3.append(nm_avg)
        num4.append(nrr_avg)
        num5.append(ndr_avg)
        num6.append(nmr_avg)

    print('tr', time1)
    print('td', time2)
    print('tm', time3)
    print('trr', time4)
    print('tdr', time5)
    print('tmr', time6)

    print('nr', num1)
    print('nd', num2)
    print('nm', num3)
    print('nrr', num4)
    print('ndr', num5)
    print('nmr', num6)

    v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.plot(v, time1)
    plt.plot(v, time2)
    plt.plot(v, time3)
    plt.plot(v, time4)
    plt.plot(v, time5)
    plt.plot(v, time6)
    # plt.plot(history.history['val_loss'])
    plt.title('schedule_delay' + str(pro))
    plt.ylabel('delay')
    plt.xlabel('vm_num')
    plt.legend(['real', 'DNN', 'matrix', 'random_real', 'random_DNN', 'random_matrix'], loc='lower left')
    plt.savefig(path + '/delay_' + str(pro) + '.png')
    plt.show()

    plt.plot(v, num1)
    plt.plot(v, num2)
    plt.plot(v, num3)
    plt.plot(v, num4)
    plt.plot(v, num5)
    plt.plot(v, num6)
    # plt.plot(history.history['val_loss'])
    plt.title('schedule_offload')
    plt.ylabel('offload')
    plt.xlabel('vm_num')
    plt.legend(['real', 'DNN', 'matrix', 'random_real', 'random_DNN', 'random_matrix'], loc='upper left')
    plt.savefig(path + '/offload_' + str(pro) + '.png')
    plt.show()
