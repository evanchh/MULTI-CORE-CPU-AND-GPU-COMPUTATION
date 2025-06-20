#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 1) 定義參數
    double alpha = 1.1;    // 例如
    double beta  = 0.4;
    double gamma = 0.4;
    double delta = 0.1;

    // 2) 初始條件與時間參數
    double x = 10.0;  // x(0)
    double y = 10.0;   // y(0)
    double t = 0.0;   // 初始時間
    double T = 20.0;  // 總模擬時間
    double dt = 0.001; // 時間步長
 
    // 3) 開檔，把結果寫到 lotka_volterra_output.txt
    FILE *fp = fopen("lotka_volterra_output.txt", "w");
    if (fp == NULL) {
        printf("Cannot open file.\n");
        return -1;
    }

    // 寫檔案標題列
    fprintf(fp, "t x y\n");

    // 4) 使用 Euler method 迭代
    while (t <= T) {
        // 每個時間步都把 t, x, y 寫進檔案
        fprintf(fp, "%f %f %f\n", t, x, y);

        // 計算 dx/dt, dy/dt
        double dxdt = alpha*x - beta*x*y;
        double dydt = -gamma*y + delta*x*y;

        // Euler 法更新 x, y
        x += dt * dxdt;
        y += dt * dydt;
        t += dt;
    }

    fclose(fp);
    printf("Simulation finished. Data saved to lotka_volterra_output.txt\n");
    return 0;
} 
