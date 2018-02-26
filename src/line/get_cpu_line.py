# 挑选分辨率是1080p，内存4g，人脸数为0、2、4的图片。3张图片，序号分别是10、130、250。其中满足条件的表是15.16.17.18.19.20.21
import matplotlib.pyplot as plt
import pandas as pd


def get_line(index):
    tet = []
    # 读取4个表
    df1 = pd.read_excel("./data2/raw/result15.xlsx", header=0)
    df2 = pd.read_excel("./data2/raw/result16.xlsx", header=0)
    df3 = pd.read_excel("./data2/raw/result17.xlsx", header=0)
    df4 = pd.read_excel("./data2/raw/result18.xlsx", header=0)
    df5 = pd.read_excel("./data2/raw/result19.xlsx", header=0)
    df6 = pd.read_excel("./data2/raw/result20.xlsx", header=0)
    df7 = pd.read_excel("./data2/raw/result21.xlsx", header=0)

    t1 = df1['frame_process_time'][index]
    t2 = df2['frame_process_time'][index]
    t3 = df3['frame_process_time'][index]
    t4 = df4['frame_process_time'][index]
    t5 = df5['frame_process_time'][index]
    t6 = df6['frame_process_time'][index]
    t7 = df7['frame_process_time'][index]

    tet.append(t1)
    tet.append(t2)
    tet.append(t3)
    tet.append(t4)
    tet.append(t5)
    tet.append(t6)
    tet.append(t7)

    return tet


if __name__ == '__main__':
    cpu = [1, 2, 3, 4, 5, 6, 7]
    for i in range(60):
        # 根据图片序号取信息
        tet1 = get_line(i)
        tet2 = get_line(i+120)
        tet3 = get_line(i+240)
        print(tet1)


        # 作图
        # summarize history for accuracy
        plt.plot(cpu, tet1)
        plt.plot(cpu, tet2)
        plt.plot(cpu, tet3)
        plt.title('tet for cpu')
        plt.ylabel('TET')
        plt.xlabel('cpu')
        plt.legend(['0', '2', '4'], loc='upper left')
        # plt.show()
        plt.savefig('./data2/line2_cpu/'+str(i)+".jpg")
        plt.savefig('./data2/line2_cpu/'+str(i) + ".eps")
        # plt.close()

