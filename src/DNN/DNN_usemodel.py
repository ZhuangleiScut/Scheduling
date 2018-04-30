import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import load_model

"""
加载模型的案例
"""

# 读取原始的数据集
df = pd.read_csv("./data/data2.csv", header=0)
# 把数据转为float类型
# df['displacement'] = df['displacement'].astype(float)

# 逐列获取数据集
# First and last (mpg and car names) are ignored for X
X = df[df.columns[0:8]]
# print(df.columns[0:3])
y = df['process_time']


# 分离数据集，将数据集按比例分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.45)

# Scale the data for 收敛优化
scaler = preprocessing.StandardScaler()

# Set the transform parameters
X_train = scaler.fit_transform(X_train)

# Build a 2 layer fully connected DNN with 10 and 5 units respectively


# 加载模型,旧版本是load_model()
model = load_model('model.h5')

# 打印模型
model.summary()

score = model.evaluate(X_test, y_test, verbose=1)
# print('loss:', score[0])
print('accuracy:', score)

# 预测
pre = model.predict(X_test, verbose=1)
# print(pre)
# print(y_test)
erro = [abs(x-y) for x, y in zip(y_test, pre)]
print(pre)
# print(y_test, pre, erro)
print('max:',max(erro))
print('min:',min(erro))
# 归一化之后的误差
erro_deal = []
for e in erro:
    erro_norm = (e-min(erro))/(max(erro)-min(erro))
    erro_deal.append(erro_norm)
print('归一化误差：', sum(erro_deal)/len(erro_deal))
print(sum(erro_deal))
