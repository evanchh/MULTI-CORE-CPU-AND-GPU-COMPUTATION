#include <stdio.h>
#include <stdlib.h>

// 根據前次作業要求或課程內容，自訂此函式
void Compute_Accelerations(double x1, double x2, double x3,
                           double v1, double v2, double v3,
                           double *a1, double *a2, double *a3,
                           double k, double M)
{
    // 以下僅用假想邏輯示意
    double k1 = k;
    double k2 = 2.0 * k;
    double k3 = 3.0 * k;
    // 例如：左邊第一個質量受力 = -k1*x1 + k2*(x2 - x1) ...等
    // (這只是示意，請依實際推導)
    *a1 = ( -k1 * x1 + k2 * (x2 - x1) ) / M;
    *a2 = ( -k2 * (x2 - x1) + k3 * (x3 - x2) ) / M;
    *a3 = ( -k3 * (x3 - x2) ) / M;
}

int main(void) {
    // 1) 初始化參數、初始位置 / 速度
    double k = 10.0;
    double M = 1.0;
    double x1 = 0.0, x2 = 0.0, x3 = 0.1;  // 例：稍微位移
    double v1 = 0.0, v2 = 0.0, v3 = 0.0;
    double t = 0.0, dt = 0.0001, T = 5.0;

    // 2) 開檔寫結果 寫到mass_spring_output.txt
    FILE *fp = fopen("mass_spring_output.txt", "w");
    fprintf(fp, "t x1 x2 x3 v1 v2 v3 a1 a2 a3\n");

    // 3) 時間迴圈 (Euler 數值積分)
    while (t <= T) {
        // (a) 計算加速度
        double a1, a2, a3;
        Compute_Accelerations(x1, x2, x3, v1, v2, v3, &a1, &a2, &a3, k, M);

        // (b) 輸出目前狀態
        fprintf(fp, "%f %f %f %f %f %f %f %f %f %f\n",
                t, x1, x2, x3, v1, v2, v3, a1, a2, a3);

        // (c) Euler 法更新速度
        v1 = v1 + dt * a1;
        v2 = v2 + dt * a2;
        v3 = v3 + dt * a3;

        // (d) Euler 法更新位置
        x1 = x1 + dt * v1;
        x2 = x2 + dt * v2;
        x3 = x3 + dt * v3;

        // (e) 更新時間
        t += dt;
    }

    fclose(fp);
    printf("Mass-spring simulation finished.\n");
    return 0;
}

