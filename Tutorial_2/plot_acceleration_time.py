import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("mass_spring_output.txt", skiprows=1) 
# skiprows=1 是因為第一行是標題

t  = data[:,0]  # 第 1 欄
a1 = data[:,7]  # 第 8 欄
a2 = data[:,8]  # 第 9 欄
a3 = data[:,9]  # 第 10 欄

plt.plot(t, a1, label='a1')
plt.plot(t, a2, label='a2')
plt.plot(t, a3, label='a3')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.savefig('acc_plot.png', dpi=300)
plt.show()

