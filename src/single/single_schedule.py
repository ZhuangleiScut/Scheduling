import os
import pandas as pd
import time
from pandas import DataFrame
from xlwt import Workbook
import matplotlib.pyplot as plt


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
############################################################################################


# 根据任务列表进行调度，尽量跟原始数据表进行关联以减少数据耦合
def get_load(task_num, equip_num, cpu_max, mem_max, MEM, input_path, output_path, PORPRO, schedule_type):
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    for i in range(task_num):
        sheet1.write(0, i + 1, 'task' + str(i))
    for t in range(equip_num):
        sheet1.write(t + 1, 0, t)
    sheet1.write(equip_num + 2, 0, equip_num + 2)
    sheet1.write(equip_num + 3, 0, equip_num + 3)
    sheet1.write(equip_num + 4, 0, equip_num + 4)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(output_path + '/load.xls')

    for i in range(equip_num):
        # 打开Excel文件
        data = pd.read_excel(input_path + '/TET_' + schedule_type + '.xlsx', sheetname=0)
        load_record = pd.read_excel(output_path + '/load.xls')
        # data_size = len(data['frame_process_time'])

        cpu = int((i / len(MEM)) + 1)
        mem = MEM[i % len(MEM)]
        print(cpu, mem)

        for img in range(task_num):
            tet = data[str(cpu) + '-' + str(mem)][img]
            load = ((PORPRO * cpu / cpu_max) + ((1 - PORPRO) * mem / mem_max)) * tet

            print(img, load)
            load_record['task' + str(img)][i] = load
            # 将更新写到新的Excel中
            DataFrame(load_record).to_excel(output_path + '/load.xls')


def get_deadline(task_num, equip_num, MEM, proportion, input_path, output_path, schedule_type):
    # 打开Excel文件
    data = pd.read_excel(input_path + '/TET_' + schedule_type + '.xlsx')
    for temp in range(task_num):
        heap = []
        # print(task_list[temp])
        for i in range(equip_num):
            # 构造一个数组
            heap.append(data[str(int((i / len(MEM)) + 1)) + '-' + str(MEM[i % len(MEM)])][temp])
            i += 1

        # 对数组进行堆排序
        heapsort(heap)

        data['deadline'][temp] = heap[proportion]
        # 将更新写到新的Excel中
        DataFrame(data).to_excel(output_path + '/deadline.xls')
    temp += 1


def get_minload(task_num, equip_num, output_path):
    data = pd.read_excel(output_path + '/load.xls')
    deadline = pd.read_excel(output_path + '/deadline.xls')

    index = []

    # 求任务的最小的负载
    for i in range(task_num):
        min_load = min(data['task' + str(i)])
        data['task' + str(i)][equip_num + 3] = min_load
        ind = 0
        for f in range(equip_num):
            k = data['task' + str(i)][f]
            if k == min_load:
                ind = f
                index.append(f)
        data['task' + str(i)][equip_num + 2] = ind

        # 把deadline也加进来
        data['task' + str(i)][equip_num + 1] = deadline['deadline'][i]

        # 将更新写到新的Excel中
        DataFrame(data).to_excel(output_path + '/min_load.xls')


def get_weight(task_num, equip_num, output_path):
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
    book.save(output_path + '/weight.xls')

    data = pd.read_excel(output_path + '/min_load.xls')
    weight = pd.read_excel(output_path + '/weight.xls')
    # 求任务的最小的负载
    for i in range(task_num):
        deadline = data['task' + str(i)][equip_num + 1]
        load = data['task' + str(i)][equip_num + 3]
        weight['task'][i] = i
        weight['deadline'][i] = deadline
        weight['min_load'][i] = load
        weight['weight'][i] = 1.0 / (deadline * load)
        weight['min_load_id'][i] = data['task' + str(i)][equip_num + 2]

        # 将更新写到新的Excel中
        DataFrame(weight).to_excel(output_path + '/weight.xls')


def get_weight_sorted(task_num, output_path):
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
    book.save(output_path + '/weight_sorted.xls')

    weight = []
    # 根据权重对任务进行排序
    data = pd.read_excel(output_path + '/weight.xls')
    dt = pd.read_excel(output_path + '/weight_sorted.xls')

    for i in range(task_num):
        w = data['weight'][i]
        weight.append(w)

    weight.sort()
    weight.reverse()

    for temp in range(task_num):
        t = data['weight'][temp]
        ind = weight.index(t)

        data['sort'][temp] = ind
        # 将更新写到新的Excel中
        DataFrame(data).to_excel(output_path + '/weight.xls')
        print(temp)
    for n in range(task_num):
        sort = data['sort'][n]

        dt['task'][sort] = data['task'][n]
        dt['deadline'][sort] = data['deadline'][n]
        dt['min_load'][sort] = data['min_load'][n]
        dt['weight'][sort] = data['weight'][n]
        dt['min_load_id'][sort] = data['min_load_id'][n]
        dt['sort'][sort] = data['sort'][n]
        print("n:", n)
    DataFrame(dt).to_excel(output_path + '/weight_sorted.xls')


##############################################################################################################
def get_min_load(result, PORPRO, cpu_max, mem_max, equips):
    """
    :param task: 任务列表
    :param result: 符合要求的配置的列表
    :param PORPRO: CPU所占比例（默认0.5）
    :param cpu_max: 一台虚拟机最多的CPU数
    :param mem_max: 一台虚拟机最多的内存数
    :param equips: 虚拟机配置表
    :return: 返回最适的id
    """
    # 获取result长度
    length = len(result)
    temp = 0
    loads = []

    for s in range(length):
        res = result[s]
        equip = equips[res]
        load = (PORPRO * equip[0] / cpu_max + (1 - PORPRO) * equip[1] / mem_max)
        loads.append(load)
    print('符合要求的负载：', loads)

    min_load = min(loads)
    index = loads.index(min_load)

    print('符合要求的最小的序号：', index)
    print(result)
    print('min_load_index', result[index])
    return result[index]


# 每一次调度
def get_scheduling(vm, task_num, MEM, equips, cpu_max, mem_max, PORPRO, input_path, output_path, schedule_type):
    """
    :param vm: 虚拟机数
    :param task_num: 任务数
    :param MEM: 内存具体分布
    :param equips: 虚拟机配置
    :param cpu_max: CPU最大数
    :param mem_max: 内存最大数
    :param PORPRO: CPU占比
    :param output_path: 输出路径
    :return:
    """
    resourse = [vm * cpu_max, vm * mem_max]
    # 打开文件
    data = pd.read_excel(output_path + '/weight_sorted.xls')
    success = 0
    fail = 0
    times = []
    # 记录功耗信息
    energys = []
    # 每个任务的调度
    for k in range(task_num):
        result = []
        weight = data['sort'][k]
        task = data['task'][k]
        deadline = data['deadline'][k]
        print(' - ' + schedule_type + '-')
        print(' - vm:' + str(vm))
        print(' - 任务：', k, task, deadline)

        # 根据task从初始数据中查TET是否符合deadline要求
        for t in range(len(equips)):
            raw = pd.read_excel(input_path + '/TET_' + schedule_type + '.xlsx')
            tet = raw[str(int((t / len(MEM)) + 1)) + '-' + str(MEM[t % len(MEM)])][int(task)]
            print('tet', tet)
            if tet <= deadline:
                result.append(t)
                # print('t', t, tet)
        print('符合要求的配置的序号：', result)

        # 如果没有符合要求的配置
        if len(result) == 0:
            # 调度失败
            data['offload'][k] = 0
            # given资源记录
            data['cpu_given'][k] = 0
            data['mem_given'][k] = 0
            times.append(deadline)
            print('没有符合要求', times)
            # 将更新写到新的Excel中
            DataFrame(data).to_excel(output_path + '/schedule/' + str(vm) + '.xlsx')
            continue

        # 根据result求负载最小的配置对应的表（要加一）
        index = get_min_load(result, PORPRO, cpu_max, mem_max, equips)
        print(' - 符合要求的最小的序号：', index)

        print('resourse:', resourse[0], resourse[1])
        print('need', equips[index][0], equips[index][1])
        data['cpu_need'][k] = equips[index][0]
        data['mem_need'][k] = equips[index][1]
        # 如果资源足够，可以进行调度。剩余资源减去调度资源
        if (resourse[0] >= equips[index][0]) and (resourse[1] >= equips[index][1]):

            resourse[0] = resourse[0] - equips[index][0]
            resourse[1] = resourse[1] - equips[index][1]

            # task 和index,这里用的是真实值
            table = pd.read_excel(input_path + '/TET_real.xlsx')
            tim = raw[str(int((index / len(MEM)) + 1)) + '-' + str(MEM[index % len(MEM)])][int(task)]
            times.append(tim)
            print(tim)

            # 调度成功
            data['offload'][k] = 1
            # given资源记录
            data['cpu_given'][k] = equips[index][0]
            data['mem_given'][k] = equips[index][1]
            # 功率为硬件*时间
            energy = (equips[index][0] / cpu_max) * tim
            energys.append(energy)
            success += 1
        else:
            fail += 1
            times.append(1.5 * deadline)

            data['offload'][k] = 0
            # given资源记录
            data['cpu_given'][k] = 0
            data['mem_given'][k] = 0
            # 将更新写到新的Excel中
            DataFrame(data).to_excel(output_path + '/schedule/' + str(vm) + '_' + schedule_type + '.xlsx')
            continue

        print('')
        # 将更新写到新的Excel中
        DataFrame(data).to_excel(output_path + '/schedule/' + str(vm) + '_' + schedule_type + '.xlsx')

    times_avg = 0
    if len(times) > 0:
        times_avg = sum(times) / len(times)

    energys_avg = 0
    if len(energys) > 0:
        energys_avg = sum(energys) / len(energys)

    print('时间：', times)
    print('时间：', times_avg)
    print('success,fail', success, fail)
    return success, times_avg, energys_avg


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('---new folder:', path)


def schedule(task_num, equips, cpu_max, mem_max, propor, MEM, PORPRO, vm_num, input_path, output_path, schedule_type):
    """
    调度程序入口
    :param task_num: 任务数
    :param equips: 配置列表
    :param cpu_max: 虚拟机最大CPU数
    :param mem_max: 虚拟机最大内存数
    :param propor: deadline定义的比例
    :param MEM:内存列表
    :param PORPRO:CPU占比
    :param vm_num:虚拟机资源数
    :param input_path:输入路径
    :param output_path:输出路径
    :param schedule_type:调度模式，分real，DNN，matrix
    :return:无返回值，运行结果是输出路径中生成的文件
    """

    output_path = output_path + '/' + schedule_type
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + '/schedule'):
        os.mkdir(output_path + '/schedule')
    ##################################################
    proportion = int(len(equips) * propor)
    # 根据任务列表求负载
    get_load(task_num, len(equips), cpu_max, mem_max, MEM, input_path, output_path, PORPRO, schedule_type)
    # 从time_matrix中获取deadline,按照比例
    get_deadline(task_num, len(equips), MEM, proportion, input_path, output_path, schedule_type)
    get_minload(task_num, len(equips), output_path)
    get_weight(task_num, len(equips), output_path)
    get_weight_sorted(task_num, output_path)
    # ##################################################
    successes = []
    times = []
    energys = []
    v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 构造调度结果表
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    sheet1.write(0, 1, "success")
    sheet1.write(0, 2, "time")
    sheet1.write(0, 3, "energy")
    sheet1.write(0, 4, "app1")
    sheet1.write(0, 5, "app2")
    for i in range(15):
        sheet1.write(i + 1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save(output_path + '/scheduling_' + schedule_type + '.xls')

    data = pd.read_excel(output_path + '/scheduling_' + schedule_type + '.xls')
    for vm in range(1, vm_num+1):
        success, _time, energy = get_scheduling(vm, task_num, MEM, equips, cpu_max, mem_max, PORPRO, input_path, output_path, schedule_type)
        successes.append(success)
        times.append(_time)
        energys.append(energy)
        data['success'][vm - 1] = success
        data['time'][vm - 1] = _time
        data['energy'][vm - 1] = energy
    print(' - success', successes)
    print(' - time', times)
    print(' - energy', energys)
    DataFrame(data).to_excel(output_path + '/scheduling_' + schedule_type + '.xls')

    plt.plot(v, successes)
    plt.title(schedule_type + '_num_schedule')
    plt.ylabel('num_schedule')
    plt.xlabel('num_vm')
    plt.legend(['num'], loc='upper right')
    plt.savefig(output_path + '/' + schedule_type + '_success.png')
    # plt.show()

    # # summarize history for accuracy
    plt.plot(v, times)
    # plt.plot(history.history['val_loss'])
    plt.title(schedule_type + '_time')
    plt.ylabel('time_schedule')
    plt.xlabel('num_vm')
    plt.legend(['time'], loc='upper right')
    plt.savefig(output_path + '/' + schedule_type + '_time.png')
    # plt.show()

    # # summarize history for accuracy
    plt.plot(v, energys)
    # plt.plot(history.history['val_loss'])
    plt.title(schedule_type + '_energys')
    plt.ylabel('energys')
    plt.xlabel('num_vm')
    plt.legend(['energys'], loc='upper left')
    plt.savefig(output_path + '/' + schedule_type + '_energy.png')
    plt.show()


if __name__ == '__main__':
    task_num = 30  # 任务数量

    vm_num = 10  # 虚拟机数量
    cpu_max = 4  # 虚拟机CPU
    mem_max = 8  # 虚拟机内存

    # deadline的比例
    propor = 0.6

    MEM = [1, 2, 3, 4, 8]
    CPU = 4

    # 具体的TET对应的配置
    equips = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 8],
              [2, 1], [2, 2], [2, 3], [2, 4], [2, 8],
              [3, 1], [3, 2], [3, 3], [3, 4], [3, 8],
              [4, 1], [4, 2], [4, 3], [4, 4], [4, 8]]
    t1 = time.time()
    input_path = './input'
    output_path = './output'
    # 判断是否存在输入输出目录
    if (not os.path.exists('./input')) or len(os.listdir('./input')) == 0:
        print('程序没有输入！')
        os.mkdir('./input')
        exit()
    if not os.path.exists('./output'):
        os.mkdir('./output')
    schedule(task_num, equips, cpu_max, mem_max, propor, MEM, 0.5, vm_num, input_path, output_path, 'real')
    t2 = time.time()
    print(t2-t1)
