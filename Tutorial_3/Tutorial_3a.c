#include <stdio.h>
#include <stdlib.h>

#define N    500
#define T    100
#define PHI  0.1

// 全域陣列
double a[N], b[N];

// Init 函式：根據流程圖初始化 a[i]
void Init() {
    int i = 0;
    while(i < N) {
        if(i < 0.5 * N) {
            a[i] = 0.0;
        } else {
            a[i] = 1.0;
        }
        i++;
    }
}

// Calc 函式：依流程圖計算 b[i]，再回寫到 a[i]
void Calc() {
    int i = 0;

    // 先計算 b[i]
    while(i < N) {
        double L, R;

        // 根據流程圖，判斷 L
        if(i == 0) {
            // i=0 => L = a[i]
            L = 0;
        } else {
            // 否則 => L = a[i-1]
            L = a[i-1];
        }

        // 根據流程圖，判斷 R
        if(i == N - 1) {
            // i=N-1 => R = a[i]
            R = 1;
        } else {
            // 否則 => R = a[i+1]
            R = a[i+1];
        }

        // b[i] = a[i] + PHI * (L + R + 2 * a[i])
        b[i] = a[i] + PHI * (L + R - 2.0 * a[i]);

        i++;
    }

    // 再把 b[i] 更新回 a[i]
    i = 0;
    while(i < N) {
        a[i] = b[i];
        i++;
    }
}

// Save 函式：輸出最終結果到檔案
void Save() {
    FILE *fp = fopen("result.txt", "w");
    if(!fp) {
        printf("Error opening file to save data.\n");
        return;
    }
    for(int i = 0; i < N; i++) {
	printf("a[%d] = %g\n",i,a[i]);
        fprintf(fp, "%f\n", a[i]);
    }
    fclose(fp);
}

// main：依照流程圖執行 Init、重複 Calc T 次、最後 Save
int main() {
    // 初始化
    Init();

    // 迭代 T 次
    int i = 0;
    while(i < T) {
        Calc();
        i++;
    }

    // 儲存結果
    Save();

    return 0;
}

