import pandas as pd
from pandas import DataFrame
from xlwt import Workbook
# import numpy as np
import random
import matplotlib.pyplot as plt
import os

"""
matrix调度
"""
############################################################################
# 在堆中做结构调整使得父节点的值大于子节点
def max_heapify(heap, heap_size, root):
    left = 2 * root + 1
    right = left + 1
    larger = root
    if left < heap_size and heap[larger] < heap[left]:
        larger = left
    if right < heap_size and heap[larger] < heap[right]:
        larger = right

    # 如果做了堆调整则larger的值等于左节点或者右节点的，这个时候做对调值操作
    if larger != root:
        heap[larger], heap[root] = heap[root], heap[larger]
        max_heapify(heap, heap_size, larger)


def build_max_heap(heap):  # 构造一个堆，将堆中所有数据重新排序
    heap_size = len(heap)  # 将堆的长度当独拿出来方便
    for i in range((heap_size - 2) // 2, -1, -1):  # 从后往前出数
        max_heapify(heap, heap_size, i)


def heapsort(heap):  # 将根节点取出与最后一位做对调，对前面len-1个节点继续进行对调整过程。
    build_max_heap(heap)
    for i in range(len(heap) - 1, -1, -1):
        heap[0], heap[i] = heap[i], heap[0]
        max_heapify(heap, i, 0)
    return heap


############################################################################################


def get_task_id(group_num, task_num):
    data = pd.read_excel('../../data/loop_partiton_VM/group' + str(group_num) + '/task_img_id_' + str(group_num) + '.xls')
    task_list = []
    for task in range(task_num):
        temp = data['data_id'][task]
        task_list.append(temp)
    return task_list


# 根据任务列表进行调度，尽量跟原始数据表进行关联以减少数据耦合
def get_load(group_num, pro, task_list, task_num):  # pro指的是比例
    path = '../../data/loop_partiton_VM/group' + str(group_num) + '/0.' + str(pro + 1) + '/matrix'
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    for i in range(task_num):
        sheet1.write(0, i + 1, 'task' + str(task_list[i]))
    for t in range(equip_num):
        sheet1.write(t + 1, 0, t)
    sheet1.write(30, 0, 29)
    sheet1.write(31, 0, 30)
    sheet1.write(32, 0, 31)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + '/load.xls')

    for i in range(equip_num):
        # 打开Excel文件
        data = pd.read_excel('../../data/raw/result' + str(i + 1) + '.xlsx')
        load_record = pd.read_excel(path + '/load.xls')
        time_record = pd.read_excel(
            '../../data/newMatrixGenerateTET/' + str(group_num) + '/matrix_predict_time_matrix_0.' + str(
                pro + 1) + '.xls')  # data_size = len(data['frame_process_time'])
        # print(data_size)
        for img in range(task_num):
            cpu = data['cpu_core'][task_list[img]]
            mem = data['mem_used'][task_list[img]]
            # tet = data['predict_time'][task_list[img]]
            # 使用预测的矩阵中的时间值
            tet = time_record['equip' + str(i)][img]
            # print(cpu,mem,tet)
            load = ((0.5 * cpu / 4) + (0.5 * mem / 8349896704)) * abs(tet)
            print(' - load', int(task_list[img]), load, cpu, mem, tet)
            load_record['task' + str(int(task_list[img]))][i] = load
            # 将更新写到新的Excel中
            DataFrame(load_record).to_excel(path + '/load.xls')


# 获取真实运行时间矩阵，以便后续求deadline
def get_time_matrix(group_num, pro, task_list, task_num):
    path = '../../data/loop_partiton_VM/group' + str(group_num) + '/0.' + str(pro + 1) + '/matrix'
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    # sheet1.write(0, 0, "id")
    for i in range(task_num):
        sheet1.write(i + 1, 0, 'task' + str(task_list[i]))
    for t in range(equip_num):
        sheet1.write(0, t + 1, 'equip' + str(t + 1))
    sheet1.write(0, 29, 'deadline')
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + '/time_matrix.xls')

    for i in range(equip_num):
        # 打开原始数据文件
        # 打开Excel文件
        dt = pd.read_excel('../../data/raw/result' + str(i + 1) + '.xlsx')
        df = pd.read_excel(path + '/time_matrix.xls')
        for t in range(task_num):
            # 这里求deadline需要用tet真实值
            df['equip' + str(i + 1)][t] = dt['frame_process_time'][task_list[t]]
            DataFrame(df).to_excel(path + '/time_matrix.xls')
        # print(i)


def get_deadline(group_num, pro, task_list, task_num, proportion):
    path = '../../data/loop_partiton_VM/group' + str(group_num) + '/0.' + str(pro + 1) + '/matrix'
    # 打开Excel文件
    data = pd.read_excel(path + '/time_matrix.xls')
    for temp in range(task_num):
        heap = []
        # print(temps)
        # print(temp)
        # print(task_list[temp])
        for i in range(equip_num):
            # 构造一个数组
            heap.append(data['equip' + str(i + 1)][temp])
            i += 1
        # 对数组进行堆排序
        # print(heap)
        # print(len(heap))
        deadline = heapsort(heap)
        # print('deadline', heap[proportion])
        data['deadline'][temp] = heap[proportion]
        # 将更新写到新的Excel中
        DataFrame(data).to_excel(path + '/deadline.xls')
    temp += 1


def get_minload(group_num, pro, task_list, task_num):
    path = '../../data/loop_partiton_VM/group' + str(group_num) + '/0.' + str(pro + 1) + '/matrix'
    data = pd.read_excel(path + '/load.xls')
    deadline = pd.read_excel(path + '/deadline.xls')

    index = []

    # 求300个任务的最小的负载
    for i in range(task_num):
        min_load = min(data['task' + str(int(task_list[i]))])
        # print(str(int(task_list[i])), min_load)
        data['task' + str(int(task_list[i]))][31] = min_load
        ind = 0
        for f in range(equip_num):
            k = data['task' + str(int(task_list[i]))][f]
            if k == min_load:
                ind = f
                index.append(f)
        data['task' + str(int(task_list[i]))][30] = ind
        # print(ind)

        # 把deadline也加进来,在29行。最小负载在31行
        data['task' + str(int(task_list[i]))][29] = deadline['deadline'][i]

        # 将更新写到新的Excel中
        DataFrame(data).to_excel(path + '/min_load.xls')
        # print(min(data['task' + str(int(task_list[i]))]))
        # print(index)


def get_weight(group_num, pro, task_list, task_num):
    path = '../../data/loop_partiton_VM/group' + str(group_num) + '/0.' + str(pro + 1) + '/matrix'
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "task")
    sheet1.write(0, 2, "cpu_need")
    sheet1.write(0, 3, "mem_need")
    sheet1.write(0, 4, "deadline")
    sheet1.write(0, 5, "min_load")
    sheet1.write(0, 6, "weight")
    sheet1.write(0, 7, "min_load_id")
    sheet1.write(0, 8, "sort")
    sheet1.write(0, 9, "cpu_given")
    sheet1.write(0, 10, "mem_given")
    sheet1.write(0, 11, "offload")
    for i in range(task_num):
        sheet1.write(i + 1, 0, i)

    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + '/weight.xls')

    data = pd.read_excel(path + '/min_load.xls')
    weight = pd.read_excel(path + '/weight.xls')
    # 求300个任务的最小的负载
    for i in range(task_num):
        deadline = data['task' + str(int(task_list[i]))][29]
        load = data['task' + str(int(task_list[i]))][31]
        weight['task'][i] = int(task_list[i])
        weight['deadline'][i] = deadline
        weight['min_load'][i] = load
        if load == 0:
            weight['weight'][i] = 1000000000 * random.random()
        else:
            weight['weight'][i] = 1.0 / (deadline * load)
        weight['min_load_id'][i] = data['task' + str(int(task_list[i]))][30]
        # print('rrrrrr',data['task' + str(i)][30])

        # 将更新写到新的Excel中
        DataFrame(weight).to_excel(path + '/weight.xls')
        # print(i)


def get_weight_sorted(group_num, pro, task_list, task_num):
    path = '../../data/loop_partiton_VM/group' + str(group_num) + '/0.' + str(pro + 1) + '/matrix'
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "task")
    sheet1.write(0, 2, "cpu_need")
    sheet1.write(0, 3, "mem_need")
    sheet1.write(0, 4, "deadline")
    sheet1.write(0, 5, "min_load")
    sheet1.write(0, 6, "weight")
    sheet1.write(0, 7, "min_load_id")
    sheet1.write(0, 8, "sort")
    sheet1.write(0, 9, "cpu_given")
    sheet1.write(0, 10, "mem_given")
    sheet1.write(0, 11, "offload")
    for i in range(task_num):
        sheet1.write(i + 1, 0, i)

    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + '/weight_sorted.xls')

    weight = []
    # 根据权重对任务进行排序
    data = pd.read_excel(path + '/weight.xls')
    dt = pd.read_excel(path + '/weight_sorted.xls')

    for i in range(task_num):
        w = data['weight'][i]
        weight.append(w)

    # print(weight)
    weight.sort()
    # print(weight)
    weight.reverse()
    # print(weight)

    # print(' - weight', weight)
    # random.shuffle(weight)
    # print(' - weight_shuffle', weight)

    # ind = weight.index(7.1535205620271753)
    # print(ind)
    for temp in range(task_num):
        t = data['weight'][temp]
        ind = weight.index(t)
        # print(ind)
        data['sort'][temp] = ind
        # 将更新写到新的Excel中
        DataFrame(data).to_excel(path + '/weight.xls')
        # print(temp)
    for n in range(task_num):
        sort = data['sort'][n]
        # print(sort)
        # dt['id'][sort] = data['id'][i]
        dt['task'][sort] = data['task'][n]
        dt['deadline'][sort] = data['deadline'][n]
        dt['min_load'][sort] = data['min_load'][n]
        dt['weight'][sort] = data['weight'][n]
        dt['min_load_id'][sort] = data['min_load_id'][n]
        dt['sort'][sort] = data['sort'][n]
    DataFrame(dt).to_excel(path + '/weight_sorted.xls')


##############################################################################################################
def get_min_load(result):
    # 获取result长度
    length = len(result)
    temp = 0
    loads = []
    for s in range(length):
        res = result[s]
        equip = equips[res]
        load = 0.5 * equip[0] / 4 + 0.5 * equip[1] / 8
        loads.append(load)
    print(' - 符合要求的负载：', loads)

    min_load = min(loads)
    index = loads.index(min_load)

    print(' - 符合要求的最小的序号：', index)
    print(' - min_load_index', result[index])
    return result[index]


def get_task_position(group_num, task, task_num):
    path = '../../data/loop_partiton_VM/group' + str(group_num)
    data = pd.read_excel(path + '/task_img_id_' + str(group_num) + '.xls')
    for i in range(task_num):
        if data['data_id'][i] == task:
            return i


# 每一次调度
def get_scheduling(group_num, pro, vm):
    path = '../../data/loop_partiton_VM/group' + str(group_num) + '/0.' + str(pro + 1) + '/matrix'

    # todo 十个虚拟机配置
    resourse = [[4, 8], [4, 8], [4, 8], [4, 8], [4, 8], [4, 8], [4, 8], [4, 8], [4, 8], [4, 8]]

    # 打开文件
    data = pd.read_excel(path + '/weight_sorted.xls')
    success = 0
    scheduled = 0
    fail = 0
    times = []
    # 记录功耗信息
    energys = []
    # 每个任务的调度
    for k in range(task_num):
        result = []
        # weight = data['sort'][k]
        task = data['task'][k]
        tet_pos = get_task_position(group_num, task, task_num)
        deadline = data['deadline'][k]
        print(' - matrix-' + 'group:' + str(group_num) + '-pro:' + str(pro))
        print(' - vm:' + str(vm))
        print(' - 任务：', k, task, deadline)

        # 根据task从初始数据中查TET是否符合deadline要求
        for t in range(28):
            # raw = pd.read_excel('../../data/scheduling_matrix/predict/0.9/result' + str(t + 1) + '.xlsx')
            # tet = raw['predict_time'][int(task)]
            # print('tet', tet)
            # if tet <= deadline:
            #     result.append(t)
            #     print('t', t, tet)
            time_record = pd.read_excel(
                '../../data/newMatrixGenerateTET/' + str(group_num) + '/matrix_predict_time_matrix_0.' + str(
                    pro + 1) + '.xls')  # data_size = len(data['frame_process_time'])
            tet = time_record['equip' + str(t)][tet_pos]
            # print('tet', tet)
            if abs(tet) <= deadline:
                result.append(t)
                # print('t', t, tet)
        print(' - 符合要求的配置的序号：', result)

        # 如果没有符合要求的配置
        if len(result) == 0:
            # 调度失败
            data['offload'][k] = 0
            # given资源记录
            data['cpu_given'][k] = 0
            data['mem_given'][k] = 0

            # 如果没有符合条件的则delay为deadline
            times.append(deadline)

            # 将更新写到新的Excel中
            DataFrame(data).to_excel(path + '/schedule/' + str(vm) + '_matrix.xlsx')
            continue

        # 根据result求负载最小的配置对应的表（要加一），index为配置序号，包括0
        index = get_min_load(result)
        print(' - 符合要求的最小的序号：', index)

        # index = random.randint(0, len(result)-1)

        print(' - resourse:', resourse[0], resourse[1])
        print(' - need', equips[index][0], equips[index][1])
        data['cpu_need'][k] = equips[index][0]
        data['mem_need'][k] = equips[index][1]

        # 求任务在index序号下的运行时间，看是否符合deadline
        # real_time = get_real_time(index)

########################################################################################################################
        # todo 先计算最适合的VM
        best_id = -1
        lest_res = 2    # 最大为2

        for res in range(vm):
            if (resourse[res][0] >= equips[index][0]) and (resourse[res][1] >= equips[index][1]):
                # todo 做差，存在数组id_vm,以便求最适vm
                res_cpu = resourse[res][0] - equips[index][0]
                res_mem = resourse[res][1] - equips[index][1]

                if lest_res > (res_cpu/4.0 + res_mem/8.0):
                    lest_res = res_cpu/4.0 + res_mem/8.0
                    best_id = res

                    # 记录max
                    if res_cpu/4.0 <= res_mem/8.0:
                        max_tuple = res_cpu/4.0
                    else:
                        max_tuple = res_mem/8.0

                    # todo
                elif lest_res == (res_cpu/4.0 + res_mem/8.0):
                    # 计算min，然后取max_min
                    if res_cpu/4.0 <= res_mem/8.0:
                        min_tuple = res_cpu/4.0
                    else:
                        min_tuple = res_mem/8.0

                    if max_tuple < min_tuple:
                        max_tuple = min_tuple
                        best_id = res

        # 如果资源足够，可以进行调度。剩余资源减去调度资源
        if best_id != -1:
            # 分配资源
            resourse[best_id][0] = resourse[best_id][0] - equips[index][0]
            resourse[best_id][1] = resourse[best_id][1] - equips[index][1]

            # task 和index,这里用的是真实值,# 求任务在index序号下的运行时间，看是否符合deadline
            table = pd.read_excel('../../data/raw/result' + str(index + 1) + '.xlsx', sheetname=0)
            time = table['frame_process_time'][task]

            # 如果真实运行时间符合deadline要求，可以成功调度
            if time <= deadline:
                times.append(time)
                # print(time)
                # 调度成功
                data['offload'][k] = 1
                # given资源记录
                data['cpu_given'][k] = equips[index][0]
                data['mem_given'][k] = equips[index][1]
                # 功率为硬件*时间
                energy = (equips[index][0] / 4) * time
                energys.append(energy)
                success += 1
                print(' - 调度成功。')
            # 调度但是不成功
            else:
                times.append(2 * deadline)
                # 调度失败
                data['offload'][k] = 2
                # given资源记录
                data['cpu_given'][k] = equips[index][0]
                data['mem_given'][k] = equips[index][1]
                # 功率为硬件*时间
                energy = (equips[index][0] / 4) * 2 * deadline
                energys.append(energy)
                print('- 调度但是不成功。')

            # 统计调度总数
            scheduled += 1
        # 没有调度
        else:
            fail += 1
            times.append(1.5*deadline)
            # 没有调度
            data['offload'][k] = 0
            # given资源记录
            data['cpu_given'][k] = 0
            data['mem_given'][k] = 0
            # 将更新写到新的Excel中
            DataFrame(data).to_excel(path + '/schedule/' + str(vm) + '_matrix.xlsx')
            continue
########################################################################################################################

        print('')
        # 将更新写到新的Excel中
        DataFrame(data).to_excel(path + '/schedule/' + str(vm) + '_matrix.xlsx')

    times_avg = 0
    if len(times) > 0:
        times_avg = sum(times) / len(times)

    energys_avg = 0
    if len(energys) > 0:
        energys_avg = sum(energys) / len(energys)

    print(' - 时间：', times)
    print(' - 时间：', times_avg)
    print(' - success,fail', success, fail)
    return success, times_avg, energys_avg, scheduled


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('---new folder:', path)


# 配置个数
equip_num = 28
# 参数配置
# vm = 10
cpu = 4
mem = 8
task_num = 30
equips = [[1, 1], [2, 1], [3, 1], [4, 1], [2, 1], [4, 1], [4, 1],
          [1, 2], [2, 2], [3, 2], [4, 2], [2, 2], [4, 2], [4, 2],
          [1, 4], [2, 4], [3, 4], [4, 4], [2, 4], [4, 4], [4, 4],
          [1, 8], [2, 8], [3, 8], [4, 8], [2, 8], [4, 8], [4, 8]]


# 调度的输入是
# task_num
# task_list
# data/predict中的预测时间
def schedule_matrix(group_num, task_num, pro):  # num是第几组实验，pro是某一组实验的第几个比例
    # 根路径
    path = '../../data/loop_partiton_VM/group' + str(group_num) + '/0.' + str(pro + 1) + '/matrix'
    if not os.path.exists(path):
        mkdir(path)
    if not os.path.exists(path + '/schedule'):
        mkdir(path + '/schedule')
    # 任务个数
    # 获取任务列表
    task_list = get_task_id(group_num, task_num)
    print(' - task_num:', task_list)

    ##################################################
    proportion = int(30 * 0.6)
    # 根据任务列表求负载
    get_load(group_num, pro, task_list, task_num)
    get_time_matrix(group_num, pro, task_list, task_num)
    # 从time_matrix中获取deadline,按照比例
    get_deadline(group_num, pro, task_list, task_num, proportion)
    get_minload(group_num, pro, task_list, task_num)
    get_weight(group_num, pro, task_list, task_num)
    get_weight_sorted(group_num, pro, task_list, task_num)
    ##################################################
    # 虚拟机个数
    # vm = 10
    successes = []
    schedule_pro = []
    times = []
    energys = []
    v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # 构造调度结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "success")
    sheet1.write(0, 2, "time")
    sheet1.write(0, 3, "energy")  # 功耗
    sheet1.write(0, 4, "schedule_pro")
    for i in range(15):
        sheet1.write(i + 1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(path + '/scheduling_matrix.xls')

    data = pd.read_excel(path + '/scheduling_matrix.xls')
    for vm in range(1, 11):
        success, time, energy, scheduled = get_scheduling(group_num, pro, vm)

        successes.append(success)
        times.append(time)
        energys.append(energy)
        if scheduled != 0:
            schedule_pro.append(float(success / scheduled))
            data['schedule_pro'][vm - 1] = float(success / scheduled)
        else:
            schedule_pro.append(0)
            data['schedule_pro'][vm - 1] = 0
        data['success'][vm - 1] = success
        data['time'][vm - 1] = time
        data['energy'][vm - 1] = energy

    print(' - success', successes)
    print(' - time', times)
    print(' - energy', energys)
    print(' - schedule_pro', schedule_pro)
    DataFrame(data).to_excel(path + '/scheduling_matrix.xls')
    # success.append(get_scheduling(vm))

    # summarize history for accuracy
    plt.plot(v, successes)
    # plt.plot(history.history['val_loss'])
    plt.title('matrix_num_schedule')
    plt.ylabel('num_schedule')
    plt.xlabel('num_vm')
    plt.legend(['num'], loc='upper left')
    plt.savefig(path + '/matrix_success.png')
    plt.show()

    # # summarize history for accuracy
    plt.plot(v, times)
    # plt.plot(history.history['val_loss'])
    plt.title('matrix_time')
    plt.ylabel('time_schedule')
    plt.xlabel('num_vm')
    plt.legend(['time'], loc='upper left')
    plt.savefig(path + '/matrix_time.png')
    plt.show()

    # # summarize history for accuracy
    plt.plot(v, energys)
    # plt.plot(history.history['val_loss'])
    plt.title('matrix_energys')
    plt.ylabel('energys')
    plt.xlabel('num_vm')
    plt.legend(['energys'], loc='upper left')
    plt.savefig(path + '/matrix_energys.png')
    plt.show()

    plt.plot(v, schedule_pro)
    # plt.plot(history.history['val_loss'])
    plt.title('matrix_schedule_pro')
    plt.ylabel('schedule_pro')
    plt.xlabel('num_vm')
    plt.legend(['schedule_pro'], loc='upper left')
    plt.savefig(path + '/matrix_schedule_pro.png')
    plt.show()
