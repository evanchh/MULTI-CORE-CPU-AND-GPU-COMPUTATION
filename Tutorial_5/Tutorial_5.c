#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 100           // 矩陣大小 100*100
#define ITER 10000


void read_matrix(const char *filename, double *A, int nrows, int ncols) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("無法開啟矩陣檔案");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (fscanf(fp, "%lf", &A[i*ncols + j]) != 1) {
                fprintf(stderr, "讀取矩陣失敗 at [%d][%d]\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(fp);
}


void read_vector(const char *filename, double *B, int n) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("無法開啟向量檔案");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%lf", &B[i]) != 1) {
            fprintf(stderr, "讀取向量失敗 at index %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    fclose(fp);
}

// 定義一個矩陣向量乘法函數，矩陣以一維陣列儲存
void Matrix_Multiply(double *A, double *x, double *result, int row, int col) {
    for (int i = 0; i < row; i++) {     //這裡i = row, 另外一個row = N 的數量
        result[i] = 0.0;                
        for (int j = 0; j < col; j++) {     //這裡j = col, 另外一個col = N 的數量
            result[i] += A[i*col + j] * x[j];   //這裡result[i]=B[i]
        }
    }
}

int main() {
    omp_set_num_threads(4);

    // 分配矩陣與向量空間
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * sizeof(double));
    if (A == NULL || B == NULL) {   
        fprintf(stderr, "記憶體配置失敗\n");  
        exit(EXIT_FAILURE);
    }

    // 讀取檔案
    read_matrix("A_matrix.txt", A, N, N);
    read_vector("B_vector.txt", B, N);

    // 共軛梯度法所需的向量
    double *x = (double*)calloc(N, sizeof(double)); // 初始猜測 x = 0
    double *R = (double*)malloc(N * sizeof(double));
    double *P = (double*)malloc(N * sizeof(double));
    double *AP = (double*)malloc(N * sizeof(double));
    if (x == NULL || R == NULL || P == NULL || AP == NULL) {
        fprintf(stderr, "記憶體配置失敗\n");
        exit(EXIT_FAILURE);
    }

    // (a) 初始殘差 R = B - A*x (但 x=0 => R=B), 並初始化 P = R
    double *AX = (double*)malloc(N * sizeof(double));
    Matrix_Multiply(A, x, AX, N, N);  // 初始 x 為0, 結果應全為0
    for (int i = 0; i < N; i++) {
        R[i] = B[i] - AX[i];
        P[i] = R[i];
    }
    free(AX);

    double PTAP = 0.0;
    double RTR = 0.0;
    double RTR_new = 0.0;

    #pragma omp parallel
    {
        for (int k = 0; k < ITER; k++) {

            #pragma omp single
            {
                PTAP = 0.0;
                RTR = 0.0;
                RTR_new = 0.0;
            }

            // (1) 計算 AP = A * P
            #pragma omp for
            for (int i = 0; i < N; i++) {
                double sum = 0.0;
                for (int j = 0; j < N; j++) {
                    sum += A[i * N + j] * P[j];
                }
                AP[i] = sum;
            }

            // (2) 計算內積 P^T * AP 與 R^T * R
            #pragma omp for reduction(+:PTAP)
            for (int i = 0; i < N; i++) {
                PTAP += P[i] * AP[i];
            }

            #pragma omp for reduction(+:RTR)
            for (int i = 0; i < N; i++) {
                RTR += R[i] * R[i];
            }

            // (3)  alpha
            double alpha;
            #pragma omp single
            {
                alpha = RTR / PTAP;
            }

            // (4)  x = x + alpha * P
            #pragma omp for
            for (int i = 0; i < N; i++) {
                x[i] += alpha * P[i];
            }

            // (5)  R = R - alpha * AP
            #pragma omp for
            for (int i = 0; i < N; i++) {
                R[i] -= alpha * AP[i];
            }

            // (6) 計算新的殘差內積
            #pragma omp for reduction(+:RTR_new)
            for (int i = 0; i < N; i++) {
                RTR_new += R[i] * R[i];
            }

            // (7)  P = R + beta * P
            double beta;
            #pragma omp single
            {
                beta = RTR_new / RTR;
            }
            #pragma omp for
            for (int i = 0; i < N; i++) {
                P[i] = R[i] + beta * P[i];
            }
        }
    }

    // 輸出結果 (部分或全部結果視需求而定)
    printf("經過 %d 次迭代後的 x 為:\n", ITER);
    printf("x = [ ");
    for (int i = 0; i < N; i++) {
        printf("%f ", x[i]);
    }
    printf("]\n\n");

    free(A);
    free(B);
    free(x);
    free(R);
    free(P);
    free(AP);

    return 0;
}
