import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


if __name__ == '__main__':



    # 不同比例的offload数-app1
    num1 = [1,2,3,4,5,6,7]
    vm1 = [40.7471295,48.36609,49.976569,52.3482782,56.9549692,61.5969283,63.3803059]



    plt.plot(num1, vm1)

    plt.title('face_swap_AVG')
    plt.ylabel('tet_AVG')
    plt.xlabel('vm数')
    # plt.legend(['1', '2', '3', '4', '5','6','7'], loc='upper left')
    # plt.show()
    plt.savefig("app1-vm-tet-AVG.jpg")
    # plt.savefig("cpu&mem-app1+2.eps")
    # plt.close()
