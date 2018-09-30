import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


if __name__ == '__main__':
    vm1 = [1,2,3,4,5,6,7,8,9,10]

    # 不同比例的offload数-app1
    # offload1 = [1,3,4,5,6,8,9,10,12,13]
    # offload2 = [2,3,5,6,8,10,11,13,14,16]
    # offload3 = [2,4,6,8,10,12,14,16,18,20]
    # offload4 = [2,4,6,8,10,12,14,16,18,20]
    # offload5 = [2,4,6,8,10,12,14,16,18,20]

    # app2
    # offload1 = [2,4,7,9,12,16,18,20,22,23]
    # offload2 = [3,6,8,12,16,20,24,28,30,30]
    # offload3 = [4,7,10,14,18,22,26,30,30,30]
    # offload4 = [4,8,12,16,20,24,28,30,30,30]
    # offload5 = [4,8,12,16,20,24,28,30,30,30]

    # app1+2
    offload1 = [2,4,7,9,12,16,18,20,22,22]
    offload2 = [3,6,8,12,16,20,23,25,27,29]
    offload3 = [4,7,10,14,18,21,24,27,29,31]
    offload4 = [4,8,12,16,20,23,26,28,30,33]
    offload5 = [4,8,12,16,20,23,26,28,30,33]

    plt.plot(vm1, offload1)
    plt.plot(vm1, offload2)
    plt.plot(vm1, offload3)
    plt.plot(vm1, offload4)
    plt.plot(vm1, offload5)
    plt.title('cpu&mem-app1+2')
    plt.ylabel('offload')
    plt.xlabel('vm数')
    plt.legend(['1', '3', '5', '7', '9'], loc='upper left')
    # plt.show()
    plt.savefig("cpu&mem-app1+2.jpg")
    plt.savefig("cpu&mem-app1+2.eps")
    # plt.close()
