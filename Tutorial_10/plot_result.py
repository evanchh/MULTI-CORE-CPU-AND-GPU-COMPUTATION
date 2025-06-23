import numpy as np
import matplotlib.pyplot as plt

# 讀取你的模擬結果檔
data = np.loadtxt('results.txt')

# 解析欄位
x = data[:,0]
y = data[:,1]
body = data[:,2]
temp = data[:,3]

# 只畫碟盤本體
mask = (body == 1)
x_disk = x[mask]
y_disk = y[mask]
temp_disk = temp[mask]

plt.figure(figsize=(7,6))
sc = plt.scatter(x_disk, y_disk, c=temp_disk, s=1, cmap='jet')
plt.colorbar(sc, label='Temperature (K)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Disk Brake Temperature Distribution')
plt.axis('equal')
plt.tight_layout()
plt.savefig('result.png', dpi=300)
plt.show()