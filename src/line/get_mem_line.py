# 挑选分辨率是1080p，CPU1个2核，人脸数为0、2、4的图片。3张图片，序号分别是10、130、250。其中满足条件的表是2、9、16、23
import matplotlib.pyplot as plt
import pandas as pd


def get_line(index):
    tet = []
    # 读取4个表
    df1 = pd.read_excel("./data2/raw/result2.xlsx", header=0)
    df2 = pd.read_excel("./data2/raw/result9.xlsx", header=0)
    df3 = pd.read_excel("./data2/raw/result16.xlsx", header=0)
    df4 = pd.read_excel("./data2/raw/result23.xlsx", header=0)

    t1 = df1['frame_process_time'][index]
    t2 = df2['frame_process_time'][index]
    t3 = df3['frame_process_time'][index]
    t4 = df4['frame_process_time'][index]

    tet.append(t1)
    tet.append(t2)
    tet.append(t3)
    tet.append(t4)

    return tet


if __name__ == '__main__':
    mem = [1, 2, 4, 8]
    for i in range(60):
        # 根据图片序号取信息
        tet1 = get_line(i)
        tet2 = get_line(i+120)
        tet3 = get_line(i+240)

        # 作图
        # summarize history for accuracy
        plt.plot(mem, tet1)
        plt.plot(mem, tet2)
        plt.plot(mem, tet3)
        plt.title('tet for mem')
        plt.ylabel('TET')
        plt.xlabel('mem')
        plt.legend(['0', '2', '4'], loc='upper left')
        # plt.show()
        plt.savefig('./data2/line2/'+str(i)+".jpg")
        plt.savefig('./data2/line2/'+str(i) + ".eps")
        # plt.close()

