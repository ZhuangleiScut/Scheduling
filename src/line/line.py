import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
import mpl_toolkits.axisartist as axisartist

#创建画布
fig = plt.figure()
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)
#将绘图区对象添加到画布中
fig.add_axes(ax)
#通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
#"-|>"代表实心箭头："->"代表空心箭头
ax.axis["bottom"].set_axisline_style("-|>", size = 1.5)
ax.axis["left"].set_axisline_style("->", size = 1.5)
#通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
ax.axis["top"].set_visible(False)
ax.axis["right"].set_visible(False)

T = np.array([0, 5, 10, 15, 20, 25, 30])
power = np.array([0, 0.6, 0.83, 0.92, 0.975, 0.997,1])
xnew = np.linspace(T.min(), T.max(), 300)  # 300 represents number of points to make between T.min and T.max
axes = plt.gca()
axes.set_xlim([0,31])
axes.set_ylim([0,1.1])


plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.title('Power consumption')
plt.ylabel('精度',verticalalignment='top')
plt.xlabel('帧率')
# plt.annotate(s = r'$2x+1=%s$', xy=(30, 1),xytext=(+30,-30), xycoords='data',textcoords='offset points', fontsize=16,arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.plot([30, 30,], [0,1,], 'k--',linewidth=1.0)
plt.plot([0, 30,], [1,1,], 'k--',linewidth=1.0)

plt.plot([15, 15,], [0,0.92,], 'm:',linewidth=1.0)
plt.plot([0, 15,], [0.92,0.92,], 'm:',linewidth=1.0)

plt.plot([15,30], [0.92,1], 'ro')
plt.annotate("(%s,%s)" % (15,0.92), xy=(15,0.92), xytext=(-20, -10), textcoords='offset points')
plt.annotate("(%s,%s)" % (30,1), xy=(30,1), xytext=(-20, -10), textcoords='offset points')

power_smooth = spline(T, power, xnew)

plt.plot(xnew, power_smooth)
plt.savefig('pow_1.png')
plt.show()