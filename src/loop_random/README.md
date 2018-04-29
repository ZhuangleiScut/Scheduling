# 代码说明
loop_radom在loop的基础上实现random调度实验的对比
## 程序主入口
schedule是程序的主入口。通过调用schedule_DNN.py、schedule_real.py、schedule_matrix.py来实现调度
## 程序输入输出说明
- 1、需要输入matrix预测的矩阵
- 2、DNN程序自动预测，不需要输入
- 3、输出为data/loop文件夹中的数据
## 代码说明
- schedule_DNN：单次DNN调度
- schedule_matrix：单次matrix调度
- schedule_real：单次real调度
- test：测试
- train_model_by_random：通过随机生成DNN结构训练DNN的model，结构存在data/DNN_train的文件夹中。跟DNN/train_model_by_random
的区别是记录了训练时间。