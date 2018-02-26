import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense

# 读取原始的数据集
df = pd.read_csv("../../data/DNN_train/sample.csv", header=0)
# 把数据转为float类型
# df['displacement'] = df['displacement'].astype(float)

# 逐列获取数据集
# First and last (mpg and car names) are ignored for X,左闭右开，13个
X = df[df.columns[0:8]]
# print(df.columns[0:3])
y = df['process_time']

# f, ax1 = plt.subplots()
# # for i in range(0, 3):
# for i in range(0, 16):
#     number = 420 + i
#     ax1.locator_params(nbins=3)
#     ax1 = plt.subplot(number)
#     plt.title(list(df)[i])
#     ax1.scatter(df[df.columns[i]], y)  # Plot a scatter draw of the  datapoints
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()

# 分离数据集，将数据集按比例分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.05)

# Scale the data for 收敛优化
scaler = preprocessing.StandardScaler()

# Set the transform parameters
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# print('X', X_train)

# Build a 2 layer fully connected DNN with 10 and 5 units respectively
# 初始化
model = Sequential()
# 添加层，参数：隐藏层的节点数，第一层的输入维数，激活函数的类型
model.add(Dense(13, activation="relu", kernel_initializer="normal", input_dim=8))
model.add(Dense(13, activation="relu", kernel_initializer="normal"))
# model.add(Dense(26, activation="relu", kernel_initializer="normal"))
# model.add(Dense(26, activation="relu", kernel_initializer="normal"))
model.add(Dense(13, activation="relu", kernel_initializer="normal"))
# model.add(Dense(6, activation="relu", kernel_initializer="normal"))
model.add(Dense(1, kernel_initializer="normal"))

# 打印模型
model.summary()

# Compile the model, whith the mean squared error as a loss function
# 编译模型，定义损失函数，均方差回归问题
model.compile(loss='mse', optimizer='adam')
print("编译完成。")

# Fit the model, in 1000 epochs
# 训练模型，1000epoch  验证集
history = model.fit(X_train, y_train, epochs=50, validation_split=0, shuffle=True, verbose=2)
print("训练完成。")
# list all data in history
print(history.history.keys())

score = model.evaluate(X_test, y_test, verbose=1)
# print('loss:', score[0])
print('accuracy:', score)

# 预测
pre = model.predict(X_test, verbose=1)
print(pre)
# print(y_test)
erro = [abs(x - y) / x for x, y in zip(y_test, pre)]
# print(pre)

# # print(y_test, pre, erro)
print('max:', max(erro))
print('min:', min(erro))
# # 归一化之后的误差
# erro_deal = []
# for e in erro:
#     erro_norm = (e-min(erro))/(max(erro)-min(erro))
#     erro_deal.append(erro_norm)
# print('归一化误差：', sum(erro_deal)/len(erro_deal))

print(sum(erro) / len(erro))

model.save('../../data/DNN_train/model.h5')

# summarize history for accuracy
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
