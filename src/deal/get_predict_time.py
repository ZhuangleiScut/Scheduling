import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import load_model
from pandas import DataFrame

result_index = ['image_size', 'resolution1', 'resolution2', 'face_num', 'face_area', 'cpu_core', 'mem_total',
                'disk_available']
for table in range(28):
    # 读取原始的数据集
    df = pd.read_excel('../../data/raw/result' + str(table + 1) + '.xlsx', header=0)
    # 把数据转为float类型
    # df['displacement'] = df['displacement'].astype(float)

    # 逐列获取数据集
    # First and last (mpg and car names) are ignored for X,左闭右开，13个
    X = df[result_index][0:300]
    # print(X)
    y = df['frame_process_time'][0:300]

    # 分离数据集，将数据集按比例分为训练集和测试集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0)

    # Scale the data for 收敛优化
    scaler = preprocessing.StandardScaler()

    # Set the transform parameters
    X_train = scaler.fit_transform(X_train)

    # Build a 2 layer fully connected DNN with 10 and 5 units respectively

    # 加载模型,旧版本是load_model()
    model = load_model('model.h5')

    # 打印模型
    model.summary()

    # 预测
    pre = model.predict(X_train, verbose=1)
    print(pre)
    for t in range(300):
        df['predict_time'][t] = pre[t]
    DataFrame(df).to_excel('../../data/scheduling_DNN/predict/result' + str(table + 1) + '.xlsx')