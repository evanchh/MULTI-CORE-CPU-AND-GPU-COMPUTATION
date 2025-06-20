#include <stdio.h>

// 計算加速度
void Compute_Accelerations(double k, double M, double x1, double x2, double x3, double a[3]) {
    a[0] = (-k * x1 + 2 * k * (x2 - x1)) / M;
    a[1] = (-2 * k * (x2 - x1) + 3 * k * (x3 - x2)) / M;
    a[2] = (-3 * k * (x3 - x2)) / M;
}

int main() {
    double k = 10.0;
    double M = 1.0;
    double x1 = 1.0, x2 = 0.5, x3 = -0.5;
    double a[3];

    // 計算加速度
    Compute_Accelerations(k, M, x1, x2, x3, a);

    // 輸出結果
    printf("Accelerations:\n");
    printf("a1 = %.3f\n", a[0]);
    printf("a2 = %.3f\n", a[1]);
    printf("a3 = %.3f\n", a[2]);

    return 0;
}

