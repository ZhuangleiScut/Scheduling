# TPDS论文《Learning Driven Computation Offloading for Asymmetrically Informed Edge Computing》
# 论文连接
https://ieeexplore.ieee.org/document/8618389

# 论文概述
 - 基于学习驱动的非对称边缘场景下的任务迁移算法
 - 任务执行时间预测算法
 - 学习驱动的任务迁移算法
 
## 调度算法文档说明
一、	调度概述
调度类型：基于real调度、基于DNN调度、基于matrix调度。

二、	调度的输入
1.	基于real调度：
	task_num:任务数
	task_list:存储在task_img_id.xlsx表中的任务id列表
	raw/result.xlsx:初始测量数据
	schedule文件夹：用来存储调度情况
2.	基于DNN调度：
	task_num:任务数
	task_list:存储在task_img_id.xlsx表中的任务id列表
	predict/result.xlsx:DNN预测值
	schedule文件夹：用来存储调度情况

3.	基于matrix调度
	task_num:任务数
	task_list:存储在task_img_id.xlsx表中的任务id列表
	raw/result.xlsx:初始测量数据
	predict_time_matrix.xlsx：存储矩阵预测的值
	schedule文件夹：用来存储调度情况

三、	调度的输出
1.	基于real调度：
	schedule.xls：不同虚拟机个数的调度成功数、平均时延
	schedule/1.xls： 不同虚拟机个数的具体调度情况
2.	基于DNN调度：
	schedule.xls：不同虚拟机个数的调度成功数、平均时延
	schedule/1.xls： 不同虚拟机个数的具体调度情况

3.	基于matrix调度
	schedule.xls：不同虚拟机个数的调度成功数、平均时延
	schedule/1.xls： 不同虚拟机个数的具体调度情况

四、	具体调度情况说明
1.	基于real调度
	get_task_id(task_num)：从task_img_id.xls表中获取需要调度的任务列表并返回
	get_load(task_list, task_num)：从初始值result表中获取每个任务的负载
		负载公式：load = ((0.5 * cpu / 4) + (0.5 * mem / 8349896704)) * tet（tet指真实运行时间）
		输入：初始值result表
返回：load.xls表，记录每个任务在每种配置下的负载
	get_time_matrix(task_list, task_num)：获取真实运行时间矩阵，以便后续求deadline
输入：初始值result表
输出：time_matrix.xls表，存储每个任务在不同配置下的真实运行时间
	get_deadline(task_list, task_num, proportion)：根据time_matrix求deadline
输入：time_matrix.xls表求deadline
输出：deadline.xls表
	get_minload(task_list, task_num)：根据load.xls表求最小负载并把deadline.xls表中的deadline信息加入。
输入：load.xls表；deadling.xls表
输出：min_load.xls表，包括最小负载，最小负载的配置id，deadline
	get_weight(task_list, task_num)：根据最小负载和deadline获取权重
公式：weight['weight'][i] = 1.0 / (deadline * load)
输入：min_load.xls表
输出：weight.xls表
	get_weight_sorted(task_list, task_num)：对weight进行排序，按照权重值升序排列
输入：weight.xls表
输出：weight_sorted.xls表
2.	基于DNN调度
	get_task_id(task_num)：从task_img_id.xls表中获取需要调度的任务列表并返回
	get_load(task_list, task_num)：从初始值result表中获取每个任务的负载
		负载公式：load = ((0.5 * cpu / 4) + (0.5 * mem / 8349896704)) * tet（tet指预测时间）
输入：初始值result表（表中有DNN预测数据predict_time）（后期可以直接使用predict表）
返回：load.xls表，记录每个任务在每种配置下的负载
	get_time_matrix(task_list, task_num)：获取真实运行时间矩阵，以便后续求deadline
输入：初始值result表
输出：time_matrix.xls表，存储每个任务在不同配置下的真实运行时间
	get_deadline(task_list, task_num, proportion)：根据time_matrix求deadline
输入：time_matrix.xls表求deadline
输出：deadline.xls表
	get_minload(task_list, task_num)：根据load.xls表求最小负载并把deadline.xls表中的deadline信息加入。
输入：load.xls表；deadling.xls表
输出：min_load.xls表，包括最小负载，最小负载的配置id，deadline
	get_weight(task_list, task_num)：根据最小负载和deadline获取权重
公式：weight['weight'][i] = 1.0 / (deadline * load)
输入：min_load.xls表
输出：weight.xls表
	get_weight_sorted(task_list, task_num)：对weight进行排序，按照权重值升序排列
输入：weight.xls表

3.	基于matrix调度
	get_task_id(task_num)：从task_img_id.xls表中获取需要调度的任务列表并返回
	get_load(task_list, task_num)：从初始值result表中获取每个任务的负载
		负载公式：load = ((0.5 * cpu / 4) + (0.5 * mem / 8349896704)) * tet（tet指真实运行时间）
		输入：初始值result表；predict_time_matrix.xls表（提供预测数据）
返回：load.xls表，记录每个任务在每种配置下的负载
	get_time_matrix(task_list, task_num)：获取真实运行时间矩阵，以便后续求deadline
输入：初始值result表
输出：time_matrix.xls表，存储每个任务在不同配置下的真实运行时间
	get_deadline(task_list, task_num, proportion)：根据time_matrix求deadline
输入：time_matrix.xls表求deadline
输出：deadline.xls表
	get_minload(task_list, task_num)：根据load.xls表求最小负载并把deadline.xls表中的deadline信息加入。
输入：load.xls表；deadling.xls表
输出：min_load.xls表，包括最小负载，最小负载的配置id，deadline
	get_weight(task_list, task_num)：根据最小负载和deadline获取权重
公式：weight['weight'][i] = 1.0 / (deadline * load)
输入：min_load.xls表
输出：weight.xls表
	get_weight_sorted(task_list, task_num)：对weight进行排序，按照权重值升序排列
输入：weight.xls表


五、	调度
不同虚拟机资源下的调度。
两个地方用到了预测值：get_load需要用预测值，调度的时候选tet<deadline时


# 输入输出说明
