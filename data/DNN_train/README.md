# 文件说明
DNN_train文件主要保存DNN模型训练时的数据

## 1-9文件夹表示不同比例的训练集训练得到的模型

 - data_id.xls：指按训练集比例随机挑选出的task的id
 - data_matrix.xls：指根据task_id挑选出的的训练集原始数据
 - models_error.xls：记录各个model的误差
 - model.h5：随机调参训练出的模型
## 八个属性model文件夹表示挑选出的训练集中有八个属性的最优的model
## 根目录的model指在1-9文件夹中挑选出来的各比例最优的model

- sample.xlsx：初始样本
- train_time：记录训练时间
- 虚拟机配置说明.xlsx：虚拟机配置说明文档