#!/usr/bin/env python3
import os

def main():
    # 原始資料：(Solver, Cells, Timestep [s], No. of Steps)
    data = [
        ("CPU", "100x100", 0.02, 400_000),
        ("CPU", "200x200", 0.01, 800_000),
        ("GPU", "100x100", 0.02, 400_000),
        ("GPU", "200x200", 0.01, 800_000),
    ]

    # 互動式輸入 Time to run
    results = []
    print("請依次輸入以下各組別的執行時間 (秒)：")
    for solver, cells, dt, steps in data:
        prompt = f"  {solver} / {cells}  → "
        while True:
            try:
                rt = float(input(prompt))
                break
            except ValueError:
                print("    請輸入數值，例如 12.34")
        results.append((solver, cells, dt, steps, rt))

    # 從輸入結果計算 speedup
    cpu100 = next(r[4] for r in results if r[0]=="CPU" and r[1]=="100x100")
    gpu100 = next(r[4] for r in results if r[0]=="GPU" and r[1]=="100x100")
    cpu200 = next(r[4] for r in results if r[0]=="CPU" and r[1]=="200x200")
    gpu200 = next(r[4] for r in results if r[0]=="GPU" and r[1]=="200x200")

    speedup100 = cpu100 / gpu100 if gpu100 else float('inf')
    speedup200 = cpu200 / gpu200 if gpu200 else float('inf')

    # 準備輸出內容
    lines = ["Solver, Number of Cells, Timestep (s), No. Steps, Time to run (s)"]
    for solver, cells, dt, steps, rt in results:
        lines.append(f"{solver}, {cells}, {dt:.3f}, {steps}, {rt:.3f}")
    lines += [
        "",
        f"Speedup (CPU vs GPU) for 100x100 cells: {speedup100:.2f}",
        f"Speedup (CPU vs GPU) for 200x200 cells: {speedup200:.2f}"
    ]

    # 寫出 Speedup.txt
    out_file = "Speedup.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(lines))

    print(f"\n已將結果寫入：{os.path.abspath(out_file)}")

if __name__ == "__main__":
    main()