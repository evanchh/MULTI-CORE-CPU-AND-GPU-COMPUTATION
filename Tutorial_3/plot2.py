import numpy as np
import matplotlib.pyplot as plt

# 讀取 data.txt
data_file = "result2.txt"
data = np.loadtxt(data_file)

# 繪製數據變化
plt.plot(data, marker="o", linestyle="-", color="b")
plt.xlabel("Index (i)")
plt.ylabel("Value of a[i]")
plt.title("Data Variation")
plt.grid()

plt.xlim(-50, len(data) + 50)  # 讓範圍變大，例如從 -50 到 (N+50)

# **存檔**
plt.savefig("data_plot2.png", dpi=300)  # 存成 PNG
# plt.savefig("data_plot.jpg", dpi=300)  # 存成 JPG
# plt.savefig("data_plot.pdf")  # 存成 PDF

# 顯示圖表
plt.show()

