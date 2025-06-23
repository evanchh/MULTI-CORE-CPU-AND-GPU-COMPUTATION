#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 1. 讀檔 & 還原 2D
x, y, body, T = np.loadtxt("results.txt", unpack=True)
nx, ny       = len(np.unique(x)), len(np.unique(y))
X            = x.reshape(ny, nx)
Y            = y.reshape(ny, nx)
Body         = body.reshape(ny, nx)
Temp         = T.reshape(ny, nx)

# 2. 開 Figure
fig, ax = plt.subplots(figsize=(8,4))

# 3. 等高線 (彩色)
levels = np.linspace(np.nanmin(Temp), np.nanmax(Temp), 8)
cs = ax.contour(
    X, Y, Temp,
    levels=levels,
    cmap='jet',
    linewidths=2
)
# 線上標籤
ax.clabel(cs, fmt='%d', inline=True, fontsize=9)

# 4. 物體邊框 (黑色)
ax.contour(
    X, Y, Body,
    levels=[0.5],
    colors='k',
    linewidths=1.5
)

# 5. 建立 ScalarMappable，只用來畫 colorbar
norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
sm = mpl.cm.ScalarMappable(cmap='jet', norm=norm)
sm.set_array([])

# 6. colorbar（純漸層）
cbar = fig.colorbar(sm, ax=ax, label='Temperature (K)')
cbar.ax.tick_params(labelsize=10)

# 7. 格式化
ax.set_xlabel('Location x (m)')
ax.set_ylabel('Location y (m)')
ax.set_aspect('equal')
ax.set_title('Temperature')
plt.tight_layout()

# 8. 存檔並顯示
plt.savefig('temperature_lines_only.png', dpi=300, bbox_inches='tight')
plt.show()