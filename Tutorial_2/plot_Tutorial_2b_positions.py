import numpy as np
import matplotlib.pyplot as plt

# 讀取檔案。若第一行是標題，就用 skiprows=1
data = np.loadtxt("mass_spring_output.txt", skiprows=1)

# 第1欄是 t，第2,3,4欄分別是 x1, x2, x3
t  = data[:,0]  # time
x1 = data[:,1]
x2 = data[:,2]
x3 = data[:,3]

plt.plot(t, x1, label='x1')
plt.plot(t, x2, label='x2')
plt.plot(t, x3, label='x3')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Mass-Spring System: Positions vs Time')
plt.legend()

# 儲存為圖檔
plt.savefig("positions_plot.png", dpi=300)

# 顯示圖形視窗 (若在遠端沒視窗可省略)
plt.show()

