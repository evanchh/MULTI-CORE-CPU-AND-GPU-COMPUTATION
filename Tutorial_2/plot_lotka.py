import numpy as np
import matplotlib.pyplot as plt

# 假設輸出檔名是 lotka_volterra_output.txt
# 而且第 1 行是標題，所以 skiprows=1
data = np.loadtxt("lotka_volterra_output.txt", skiprows=1)

t = data[:,0]  # 第 1 欄
x = data[:,1]  # 第 2 欄
y = data[:,2]  # 第 3 欄

# 畫圖
plt.plot(t, x, label="Prey (x)")
plt.plot(t, y, label="Predator (y)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()

# 儲存成圖檔
plt.savefig("lotka_volterra_plot.png", dpi=300)

# 顯示圖形視窗
plt.show()

