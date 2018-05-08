import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from xlwt import Workbook
import math


"""
    获取energy数据；
    6个图：正常3个；随机3个
    画图：将上述数据画图
"""
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
    pro = 0.9
    # 个数
    group = 10

    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "vm_num")
    for a in range(10):  # 纵轴
        sheet1.write(a + 1, 0, a)
        # print(task_list[a])
    # 横轴
    sheet1.write(0, 1, 'real')
    sheet1.write(0, 2, 'DNN')
    sheet1.write(0, 3, 'matrix')
    sheet1.write(0, 4, 'random_real')
    sheet1.write(0, 5, 'random_DNN')
    sheet1.write(0, 6, 'random_matrix')
    # 保存Excel book.save('path/文件名称.xls')
    book.save('0.xls')
    print('建表完成。')

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

    var1 = []
    var2 = []
    var3 = []
    var4 = []
    var5 = []
    var6 = []
    var7 = []
    var8 = []
    var9 = []
    var10 = []
    var11 = []
    var12 = []

    data1 = pd.read_excel('0.xls')
    data2 = pd.read_excel('0.xls')

    data3 = pd.read_excel('0.xls')
    data4 = pd.read_excel('0.xls')

    data5 = pd.read_excel('0.xls')
    data6 = pd.read_excel('0.xls')

    # 10种虚拟机配置
    for v in range(10):
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

        # group 组
        for t in range(group):
            # 算10组平均值,先构造数组
            r = real[t]
            d = DNN[t]
            m = matrix[t]
            rr = real_random[t]
            dr = DNN_random[t]
            mr = matrix_random[t]

            tr.append(r['energy'][v])
            nr.append(1)

            td.append(d['energy'][v])
            nd.append(d['schedule_pro'][v])

            tm.append(m['energy'][v])
            nm.append(m['schedule_pro'][v])

            trr.append(rr['energy'][v])
            nrr.append(1)

            tdr.append(dr['energy'][v])
            ndr.append(dr['schedule_pro'][v])

            tmr.append(mr['energy'][v])
            nmr.append(mr['schedule_pro'][v])
            # print(tr,nr)

        # 计算均值
        tr_avg = sum(tr) / len(tr)
        nr_avg = sum(nr) / len(nr)
        td_avg = sum(td) / len(td)
        nd_avg = sum(nd) / len(nd)
        tm_avg = sum(tm) / len(tm)
        nm_avg = sum(nm) / len(nm)

        trr_avg = sum(trr) / len(trr)
        nrr_avg = sum(nrr) / len(nrr)
        tdr_avg = sum(tdr) / len(tdr)
        ndr_avg = sum(ndr) / len(ndr)
        tmr_avg = sum(tmr) / len(tmr)
        nmr_avg = sum(nmr) / len(nmr)

        # 计算方差
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        sum6 = 0
        sum7 = 0
        sum8 = 0
        sum9 = 0
        sum10 = 0
        sum11 = 0
        sum12 = 0
        for k in range(group):
            sum1 += math.pow(tr[k] - tr_avg, 2)
            sum2 += math.pow(td[k] - td_avg, 2)
            sum3 += math.pow(tm[k] - tm_avg, 2)
            sum4 += math.pow(trr[k] - trr_avg, 2)
            sum5 += math.pow(tdr[k] - tdr_avg, 2)
            sum6 += math.pow(tmr[k] - tmr_avg, 2)
            sum7 += math.pow(nr[k] - nr_avg, 2)
            sum8 += math.pow(nd[k] - nd_avg, 2)
            sum9 += math.pow(nm[k] - nm_avg, 2)
            sum10 += math.pow(nrr[k] - nrr_avg, 2)
            sum11 += math.pow(ndr[k] - ndr_avg, 2)
            sum12 += math.pow(nmr[k] - nmr_avg, 2)
        tr_var = sum1 / len(tr)
        td_var = sum2 / len(td)
        tm_var = sum3 / len(tm)
        trr_var = sum4 / len(trr)
        tdr_var = sum5 / len(tdr)
        tmr_var = sum6 / len(tmr)
        nr_var = sum7 / len(nr)
        nd_var = sum8 / len(nd)
        nm_var = sum9 / len(nm)
        nrr_var = sum10 / len(nrr)
        ndr_var = sum11 / len(ndr)
        nmr_var = sum12 / len(nmr)
        print(nd_var)

        # 构造数组
        var1.append(tr_var)
        var2.append(td_var)
        var3.append(tm_var)
        var4.append(trr_var)
        var5.append(tdr_var)
        var6.append(tmr_var)
        var7.append(nr_var)
        var8.append(nd_var)
        var9.append(nm_var)
        var10.append(nrr_var)
        var11.append(ndr_var)
        var12.append(nmr_var)

        data1['real'][v] = var1[v]
        data1['DNN'][v] = var2[v]
        data1['matrix'][v] = var3[v]
        data1['random_real'][v] = var4[v]
        data1['random_DNN'][v] = var5[v]
        data1['random_matrix'][v] = var6[v]

        data2['real'][v] = var7[v]
        data2['DNN'][v] = var8[v]
        data2['matrix'][v] = var9[v]
        data2['random_real'][v] = var10[v]
        data2['random_DNN'][v] = var11[v]
        data2['random_matrix'][v] = var12[v]
        DataFrame(data1).to_excel('energy_'+str(pro)+'_std.xls')
        DataFrame(data2).to_excel('schedule_pro_'+str(pro)+'_std.xls')

        # 构造数组画图
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
        data5['real'][v] = time1[v]
        data5['DNN'][v] = time2[v]
        data5['matrix'][v] = time3[v]
        data5['random_real'][v] = time4[v]
        data5['random_DNN'][v] = time5[v]
        data5['random_matrix'][v] = time6[v]

        data6['real'][v] = num1[v]
        data6['DNN'][v] = num2[v]
        data6['matrix'][v] = num3[v]
        data6['random_real'][v] = num4[v]
        data6['random_DNN'][v] = num5[v]
        data6['random_matrix'][v] = num6[v]
        DataFrame(data5).to_excel('energy_'+str(pro)+'_mean.xls')
        DataFrame(data6).to_excel('schedule_pro_'+str(pro)+'_mean.xls')

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
