from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import pandas as pd


if __name__ == "__main__":
    # 读取原始的数据集
    df = pd.read_csv("../../data/DNN_train/sample.csv", header=0)
    # 逐列获取数据集
    # First and last (mpg and car names) are ignored for X,左闭右开，13个
    x = df[df.columns[0:8]]
    y = df['process_time']
    # print(x)
    # print(y)

    # 划分训练集和测试集x表示样本特征集，y表示样本结果  test_size 样本占比,random_state 随机数的种子
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

    # Scale the data for 收敛优化
    scaler = preprocessing.StandardScaler()
    # Set the transform parameters
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # 随机森林 n_estimators：决策树的个数,越多越好,不过值越大，性能就会越差,至少100
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(x_train, y_train)
    y_rf = rf.predict(x_test)

    # Bagging估计
    bg = BaggingRegressor(DecisionTreeRegressor())
    bg.fit(x_train, y_train)
    y_bg = bg.predict(x_test)

    # print('lr', y_lr)
    print('rf', y_rf)
    print('bg', y_bg)
    print(y_test)
    print(len(y_rf))

