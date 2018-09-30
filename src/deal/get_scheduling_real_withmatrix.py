import pandas as pd
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
    return heap


############################################################################################


# 根据任务列表进行调度，尽量跟原始数据表进行关联以减少数据耦合
def get_load(task_num):
    # 构造结果表
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('Sheet 1')
    sheet1.write(0, 0, "id")
    for i in range(task_num):
        sheet1.write(0, i + 1, 'task' + str(i))
    for t in range(equip_num):
        sheet1.write(t + 1, 0, t)
    sheet1.write(task_num, 0, task_num)
    sheet1.write(task_num+1, 0, task_num+1)
    sheet1.write(task_num+2, 0, task_num+2)
    # 保存Excel book.save('path/文件名称.xls')
    book.save('./data/load.xls')


    for i in range(equip_num):
        # 打开Excel文件
        data = pd.read_excel('./data/TET.xlsx', sheetname=0)
        load_record = pd.read_excel('./data/load.xls')
        # data_size = len(data['frame_process_time'])
        # print(data_size)

        cpu = int((i / 6) +1)
        mem = MEM[i%6]
        print(cpu,mem)

        for img in range(task_num):
            # print(img,task_num)
            tet = data[str(cpu)+'-'+str(mem)][img-1]
            # print(tet)
            # print(cpu,mem,tet)
            load = ((PORPRO * cpu / 4) + ((1-PORPRO) * mem / 16)) * tet
            print('load', img, load, cpu, mem, tet)
            load_record['task' + str(img)][i] = load
            # 将更新写到新的Excel中
            DataFrame(load_record).to_excel('./data/load.xls')


def get_deadline(task_num, proportion):
    # 打开Excel文件
    data = pd.read_excel('./data/TET.xlsx')
    for temp in range(task_num):
        heap = []

        for i in range(equip_num):
            # 构造一个数组
            heap.append(data[str(int((i / 6) +1))+'-'+str(MEM[i%6])][temp])
            i += 1
        # 对数组进行堆排序
        print(heap)
        print(len(heap))
        deadline = heapsort(heap)
        print('deadline', heap[proportion])
        data['deadline'][temp] = heap[proportion]
        # 将更新写到新的Excel中
        DataFrame(data).to_excel('./data/deadline.xls')
    temp += 1


def get_minload(task_num):
    data = pd.read_excel('./data/load.xls')
    deadline = pd.read_excel('./data/deadline.xls')

    index = []

    # 求300个任务的最小的负载
    for i in range(task_num):
        min_load = min(data['task' + str(i)])
        print(str(i), min_load)
        data['task' + str(i)][task_num+1] = min_load
        ind = 0
        for f in range(equip_num):
            k = data['task' + str(i)][f]
            if k == min_load:
                ind = f
                index.append(f)
        data['task' + str(i)][task_num] = ind
        print(ind)

        # 把deadline也加进来,在29行。最小负载在31行
        data['task' + str(i)][task_num-1] = deadline['deadline'][i]

        # 将更新写到新的Excel中
        DataFrame(data).to_excel('./data/min_load.xls')
        print(min(data['task' + str(i)]))
        print(index)


def get_weight(task_num):
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
    book.save('./data/weight.xls')

    data = pd.read_excel('./data/min_load.xls')
    weight = pd.read_excel('./data/weight.xls')
    # 求300个任务的最小的负载
    for i in range(task_num):
        deadline = data['task' + str(i)][task_num-1]
        load = data['task' + str(i)][task_num+1]
        weight['task'][i] = i
        weight['deadline'][i] = deadline
        weight['min_load'][i] = load
        weight['weight'][i] = 1.0 / (deadline * load)
        weight['min_load_id'][i] = data['task' + str(i)][task_num]

        # 将更新写到新的Excel中
        DataFrame(weight).to_excel('./data/weight.xls')
        print(i)


def get_weight_sorted(task_num):
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
    book.save('./data/weight_sorted.xls')

    weight = []
    # 根据权重对任务进行排序
    data = pd.read_excel('./data/weight.xls')
    dt = pd.read_excel('./data/weight_sorted.xls')

    for i in range(task_num):
        w = data['weight'][i]
        weight.append(w)
    print(weight)
    weight.sort()
    print(weight)
    weight.reverse()
    print(weight)
    # ind = weight.index(7.1535205620271753)
    # print(ind)
    for temp in range(task_num):
        t = data['weight'][temp]
        ind = weight.index(t)
        # print(ind)
        data['sort'][temp] = ind
        # 将更新写到新的Excel中
        DataFrame(data).to_excel('./data/weight.xls')
        print(temp)
    for n in range(task_num):
        sort = data['sort'][n]
        print(sort)
        # dt['id'][sort] = data['id'][i]
        dt['task'][sort] = data['task'][n]
        dt['deadline'][sort] = data['deadline'][n]
        dt['min_load'][sort] = data['min_load'][n]
        dt['weight'][sort] = data['weight'][n]
        dt['min_load_id'][sort] = data['min_load_id'][n]
        dt['sort'][sort] = data['sort'][n]
    DataFrame(dt).to_excel('./data/weight_sorted.xls')


##############################################################################################################
def get_min_load(result):
    # 获取result长度
    length = len(result)
    temp = 0
    loads = []
    for s in range(length):
        res = result[s]
        equip = equips[res]
        load = PORPRO * equip[0] / 4 + (1-PORPRO) * equip[1] / 16
        loads.append(load)
    print('符合要求的负载：', loads)

    min_load = min(loads)
    index = loads.index(min_load)

    print('符合要求的最小的序号：', index)
    print('min_load_index', result[index])
    return result[index]


# 每一次调度
def get_scheduling(vm):
    resourse = [vm * CPU, vm * MEM[5]]
    # 打开文件
    data = pd.read_excel('./data/weight_sorted.xls')
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
        print('任务：', k, task, deadline)

        # 根据task从初始数据中查TET是否符合deadline要求
        for t in range(equip_num):
            raw = pd.read_excel('./data/TET.xlsx')
            tet = raw[str(int((t / 6) +1))+'-'+str(MEM[t%6])][k]
            print('tet', tet)
            if tet <= deadline:
                result.append(t)
                print('t', t, tet)
        print('符合要求的配置的序号：', result)

        # 如果没有符合要求的配置
        if len(result) == 0:
            # 调度失败
            data['offload'][k] = 0
            # given资源记录
            data['cpu_given'][k] = 0
            data['mem_given'][k] = 0
            times.append(deadline)
            print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm没有符合要求', times)
            # 将更新写到新的Excel中
            DataFrame(data).to_excel('./data/' + str(vm) + '.xlsx')
            continue

        # 根据result求负载最小的配置对应的表（要加一）
        index = get_min_load(result)
        print('符合要求的最小的序号：', index)

        print('resourse:', resourse[0], resourse[1])
        print('need', equips[index][0], equips[index][1])
        data['cpu_need'][k] = equips[index][0]
        data['mem_need'][k] = equips[index][1]
        # 如果资源足够，可以进行调度。剩余资源减去调度资源
        if (resourse[0] >= equips[index][0]) and (resourse[1] >= equips[index][1]):

            resourse[0] = resourse[0] - equips[index][0]
            resourse[1] = resourse[1] - equips[index][1]

            # task 和index,这里用的是真实值
            table = pd.read_excel('./data/TET.xlsx')
            tim = raw[str(int((index / 6) + 1)) + '-' + str(MEM[index % 6])][int(task)]
            times.append(tim)
            print(tim)

            # 调度成功
            data['offload'][k] = 1
            # given资源记录
            data['cpu_given'][k] = equips[index][0]
            data['mem_given'][k] = equips[index][1]
            # 功率为硬件*时间
            energy = (equips[index][0] / 4) * tim
            energys.append(energy)
            success += 1
        else:
            fail += 1
            times.append(deadline)
            print('klkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
            print(deadline)
            print(times)
            # 调度失败
            data['offload'][k] = 0
            # given资源记录
            data['cpu_given'][k] = 0
            data['mem_given'][k] = 0
            # 将更新写到新的Excel中
            DataFrame(data).to_excel('./data/' + str(vm) + '.xlsx')
            continue

        print('')
        # 将更新写到新的Excel中
        DataFrame(data).to_excel('./data/' + str(vm) + '.xlsx')

    times_avg = 0
    if len(times) > 0:
        times_avg = sum(times) / len(times)

    energys_avg = 0
    if len(energys) > 0:
        energys_avg = sum(energys) / len(energys)


    print('ttttttttttttttttttttttttttttttttttttttt',len(times))

    print('时间：', times)
    print('时间：', times_avg)
    print('success,fail', success, fail)
    return success, times_avg, energys_avg


# 配置个数
equip_num = 24
# 参数配置
vm = 10
PORPRO = 0.5
task_num = 30
MEM = [1,2,3,4,8,16]
CPU = 4

equips = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 8], [1, 16],
          [2, 1], [2, 2], [2, 3], [2, 4], [2, 8], [2, 16],
          [3, 1], [3, 2], [3, 3], [3, 4], [3, 8], [3, 16],
          [4, 1], [4, 2], [4, 3], [4, 4], [4, 8], [4, 16]]


if __name__ == '__main__':
    ##################################################
    proportion = int(equip_num * 0.6)
    # 根据任务列表求负载
    get_load(task_num)
    # 从time_matrix中获取deadline,按照比例
    get_deadline(task_num, proportion)
    get_minload(task_num)
    get_weight(task_num)
    get_weight_sorted(task_num)
    ##################################################
    # 虚拟机个数
    vm = 10
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
    for i in range(15):
        sheet1.write(i + 1, 0, i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save('./data/scheduling_real.xls')

    data = pd.read_excel('./data/scheduling_real.xls')
    for vm in range(1, 11):
        success, time, energy = get_scheduling(vm)
        successes.append(success)
        times.append(time)
        energys.append(energy)
        data['success'][vm - 1] = success
        data['time'][vm - 1] = time
        data['energy'][vm - 1] = energy
    print('success', successes)
    print('time', times)
    print('energy', energys)
    DataFrame(data).to_excel('./data/scheduling_real.xls')
    # success.append(get_scheduling(vm))

    # summarize history for accuracy
    plt.plot(v, successes)
    # plt.plot(history.history['val_loss'])
    plt.title('num_schedule')
    plt.ylabel('num_schedule')
    plt.xlabel('num_vm')
    plt.legend(['num'], loc='upper right')
    plt.savefig('./data/success.png')
    plt.show()

    # # summarize history for accuracy
    plt.plot(v, times)
    # plt.plot(history.history['val_loss'])
    plt.title('time')
    plt.ylabel('time_schedule')
    plt.xlabel('num_vm')
    plt.legend(['time'], loc='upper right')
    plt.savefig('./data/time.png')
    plt.show()

    # # summarize history for accuracy
    plt.plot(v, energys)
    # plt.plot(history.history['val_loss'])
    plt.title('energys')
    plt.ylabel('energys')
    plt.xlabel('num_vm')
    plt.legend(['energys'], loc='upper left')
    plt.savefig('./data/energy.png')
    plt.show()

