import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# -----------------------------------------------------------------------------
# 1. 讀入節點座標 (mesh_nodes.txt)，並把單位從 毫米 (mm) 換算成 公尺 (m)
#    格式：第一行是節點總數 int，接著每行兩個浮點數 x(mm) y(mm)
# -----------------------------------------------------------------------------
# === 1. 直接讀取節點座標，不含節點總數 ===
coords_mm = np.loadtxt('nodes.txt')  # 每行兩個 float：x y (mm)
coords = coords_mm / 1000.0         # 換算成公尺 (m)
num_nodes = coords.shape[0]

# -----------------------------------------------------------------------------
# 2. 找出固定節點 (radius < 21 mm)，並讀入只含自由節點的位移 (node_displacements.txt)
# -----------------------------------------------------------------------------
distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
r_fixed = 21.0 / 1000.0   # 21 mm -> 0.021 m
fixed_indices = np.where(distances < r_fixed)[0]

# 讀入只含 13940 個自由節點的位移 (dx, dy)，單位為 m
disp_free = np.loadtxt('cg_solution.csv')  # shape = (27880, 1)
dx_free = disp_free[::2]
dy_free = disp_free[1::2]

# 將自由節點的位移填回所有 13964 個節點，固定節點保持 0
dx_full = np.zeros(num_nodes)
dy_full = np.zeros(num_nodes)
all_indices = np.arange(num_nodes)
fixed_set = set(fixed_indices.tolist())
free_indices = [i for i in all_indices if i not in fixed_set]  # 共 13940 個自由節點

for k, idx in enumerate(free_indices):
    dx_full[idx] = dx_free[k]
    dy_full[idx] = dy_free[k]

# -----------------------------------------------------------------------------
# 3. 計算變形後的節點位置 new_coords（放大 10× 以便可視化）
# -----------------------------------------------------------------------------
scale_factor = 10.0
new_coords = coords + scale_factor * np.vstack((dx_full, dy_full)).T
# new_coords.shape = (13964, 2)

# -----------------------------------------------------------------------------
# 4. 讀入元素拓樸 (mesh_elements.txt)，格式：第一行元素總數 int，每行三個 0-based 節點索引
# -----------------------------------------------------------------------------
# 讀取 elements.txt（第一行是總數，接著是每行三個節點 index）
with open('elements.txt', 'r') as f:
    num_elems = int(f.readline().strip())     # 25946
    elems = np.loadtxt(f, dtype=int)          # shape = (25946, 3)

# -----------------------------------------------------------------------------
# 5. 計算每個單元的應力 σₓₓ, σᵧᵧ, σₓᵧ
# -----------------------------------------------------------------------------
E = 210e9
nu = 0.3
D = (E / (1 - nu**2)) * np.array([
    [1.0,   nu,      0.0],
    [nu,    1.0,     0.0],
    [0.0,   0.0, (1 - nu) / 2.0]
])

# 把完整節點位移組合成 U_global (長度 = 2 * num_nodes)
U_global = np.zeros(2 * num_nodes)
U_global[0::2] = dx_full
U_global[1::2] = dy_full

# 存儲每個單元的應力 (σₓₓ, σᵧᵧ, σₓᵧ)
stress_elems = np.zeros((num_elems, 3))
for i in range(num_elems):
    n1, n2, n3 = elems[i]
    x1, y1 = coords[n1]
    x2, y2 = coords[n2]
    x3, y3 = coords[n3]

    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * abs(area2)
    if A == 0:
        continue  # 跳過退化單元

    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1

    B = (1.0 / (2.0 * A)) * np.array([
        [b1,    0.0, b2,    0.0, b3,    0.0],
        [0.0,   c1,  0.0,  c2,  0.0,  c3],
        [c1,    b1,  c2,   b2,  c3,   b3]
    ])

    U_local = np.array([
        U_global[2 * n1],     U_global[2 * n1 + 1],
        U_global[2 * n2],     U_global[2 * n2 + 1],
        U_global[2 * n3],     U_global[2 * n3 + 1]
    ])
    eps = B.dot(U_local)
    sigma = D.dot(eps)
    stress_elems[i, :] = sigma

sigma_xx = stress_elems[:, 0]  # 提取 σₓₓ 分量

# -----------------------------------------------------------------------------
# 6. 繪製第一張圖：原始網格（節點+線）與變形後網格（節點+線）疊加
#    並在圖例中標示 “At rest” / “Under load”
#    保存為 overlay_nodes_edges.png
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.title("At Rest vs. Under Load (10× Deformed)")
plt.gca().set_aspect('equal')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

# 6.1 原始網格線 (藍色)
for tri in elems:
    poly = coords[tri]
    poly = np.vstack((poly, poly[0]))  # 閉合
    plt.plot(poly[:, 0], poly[:, 1], color='blue', linewidth=0.5)

# 6.2 原始節點 (藍色點)，調整 s 可控制大小
plt.scatter(coords[:, 0], coords[:, 1],
            s=1, c='blue')  # 僅繪製點，不標 label

# 6.3 變形後網格線 (黃色)
for tri in elems:
    poly2 = new_coords[tri]
    poly2 = np.vstack((poly2, poly2[0]))
    plt.plot(poly2[:, 0], poly2[:, 1], color='yellow', linewidth=0.5)

# 6.4 變形後節點 (黃色點)，調整 s 可控制大小
plt.scatter(new_coords[:, 0], new_coords[:, 1],
            s=1, c='yellow')

# 6.5 建立代理圖例句柄：藍色方塊與黃色方塊
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='blue', label='At rest')
yellow_patch = mpatches.Patch(color='yellow', label='Under load')
plt.legend(handles=[blue_patch, yellow_patch], loc='upper right')
plt.axis('off')
plt.title('Disk Displacement (Brake Stress Problem)')
plt.savefig("deformed_mesh.png", dpi=300)
plt.show()
print("已保存：'deformed_mesh.png' —— 原始網格+節點 與 變形後網格+節點 叠加")

# -----------------------------------------------------------------------------
# 7. 繪製第二張圖：純 σₓₓ 應力分布 (不顯示網格或節點)
#    保存為 stress_xx.png
# -----------------------------------------------------------------------------
x_nodes = coords[:, 0]
y_nodes = coords[:, 1]
triang = mtri.Triangulation(x_nodes, y_nodes, elems)

plt.figure(figsize=(8, 6))
tp=plt.tripcolor(triang, facecolors=sigma_xx, edgecolors='none', cmap='viridis')
cbar = plt.colorbar(tp, orientation='vertical')
cbar.set_label(r'$\sigma_{xx}$ (Pa)', fontsize=12)

plt.title(r'Element $\sigma_{xx}$ Distribution')
plt.gca().set_aspect('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.savefig("stress.png", dpi=300)
plt.show()
print("已保存：'stress.png' —— 純 σₓₓ 應力分布圖")