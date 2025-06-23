#include <stdio.h>
#include <stdlib.h>

#define N 10  
#define NN (N*N)

// 邊界條件
#define T_LEFT   0.0
#define T_RIGHT  1.0
#define T_TOP    0.0
#define T_BOTTOM 0.0

int getIndex(int i, int j) {
    return i * N + j;
}

int main() {
    double *A = calloc(NN * NN, sizeof(double));
    double *B = calloc(NN, sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = getIndex(i, j);
            A[idx * NN + idx] = 4.0;

            // left neighbor
            if (j > 0)
                A[idx * NN + getIndex(i, j - 1)] = -1.0;
            else
                B[idx] += T_LEFT;

            // right neighbor
            if (j < N - 1)
                A[idx * NN + getIndex(i, j + 1)] = -1.0;
            else
                B[idx] += T_RIGHT;

            // top neighbor
            if (i > 0)
                A[idx * NN + getIndex(i - 1, j)] = -1.0;
            else
                B[idx] += T_TOP;

            // bottom neighbor
            if (i < N - 1)
                A[idx * NN + getIndex(i + 1, j)] = -1.0;
            else
                B[idx] += T_BOTTOM;
        }
    }

    // 將 A 儲存為文字檔
    FILE *fa = fopen("A_matrix.txt", "w");
    for (int i = 0; i < NN; i++) {
        for (int j = 0; j < NN; j++) {
            fprintf(fa, "%lf ", A[i * NN + j]);
        }
        fprintf(fa, "\n");
    }
    fclose(fa);

    // 將 B 儲存為文字檔
    FILE *fb = fopen("B_vector.txt", "w");
    for (int i = 0; i < NN; i++) {
        fprintf(fb, "%lf\n", B[i]);
    }
    fclose(fb);

    printf("Matrix A and vector B saved.\n");

    free(A);
    free(B);
    return 0;
}
