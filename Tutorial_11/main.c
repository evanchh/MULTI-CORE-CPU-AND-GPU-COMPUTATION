#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"
#include <math.h> 
#include <cuda_runtime.h>

#define MAX_NNZ    2000000   // 根據 Kr_Final.txt 預估數量
#define MAX_NROWS  27880     // 因為 Kr 是 27880×27880 的稀疏矩陣

int main() {

    int *row, *col;
    float *val;

    Set_GPU_Device(1);

    // COO 格式
    row = (int*)malloc(MAX_NNZ * sizeof(int));
    col = (int*)malloc(MAX_NNZ * sizeof(int));
    val = (float*)malloc(MAX_NNZ * sizeof(float));

    // === 讀入 Kr_Final.txt ===
    FILE *fp = fopen("Kr_Final.txt", "r");
    if (!fp) {
        printf("Failed to open Kr_Final.txt\n");
        return -1;
    }
    
    int nnz = 0;
    int r, c;
    float v;
    while (fscanf(fp, "%d %d %e", &r, &c, &v) == 3) {
        row[nnz] = r - 1;
        col[nnz] = c - 1;
        val[nnz] = v;
        nnz++;
        if (nnz >= MAX_NNZ) {
            printf("Too many non-zero entries. Increase MAX_NNZ.\n");
            break;
        }
    }
    fclose(fp);
    printf("Read %d non-zero entries from Kr_Final.txt\n\n", nnz);

    // === COO -> CSR ===
    int *row_ptr = (int*)calloc((MAX_NROWS + 1), sizeof(int));
    int *col_idx = (int*)malloc(nnz * sizeof(int));
    float *val_csr = (float*)malloc(nnz * sizeof(float));

    // Step 1: 統計每列非零元素數量
    for (int i = 0; i < nnz; i++) {
        row_ptr[row[i] + 1]++;
    }

    // Step 2: 累加成 row_ptr[]
    for (int i = 0; i < MAX_NROWS; i++) {
        row_ptr[i + 1] += row_ptr[i];
    }

    // Step 3: 根據 row_ptr 將 col 與 val 排入正確位置
    int *counter = (int*)calloc(MAX_NROWS, sizeof(int));
    for (int i = 0; i < nnz; i++) {
        int r = row[i];
        int dest = row_ptr[r] + counter[r];
        col_idx[dest] = col[i];
        val_csr[dest] = val[i];
        counter[r]++;
    }
    free(counter);

    int *d_row_ptr, *d_col_idx;
    float *d_val, *d_result;
    Allocate_CSR_Memory(&d_row_ptr, &d_col_idx, &d_val, MAX_NROWS, nnz);
    Send_CSR_To_Device(&d_row_ptr, row_ptr, &d_col_idx, col_idx, &d_val, val_csr, MAX_NROWS, nnz);
    
    // // === Milestone 2 ===
    // cudaMalloc((void**)&d_result, nnz * sizeof(float));
    // float alpha = 2.0f;
    // Launch_Vector_Multiply_Constant(d_result, d_val, alpha, nnz);
    // cudaDeviceSynchronize();  // 確保 GPU 運算完成

    // // === GPU 驗證：抓回 GPU 上的 d_val[] 前 10 筆確認資料正確性 ===
    // float *val_check = (float*)malloc(10 * sizeof(float));
    // cudaMemcpy(val_check, d_result, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("\n[Check] First 10 values of val_csr before GPU:\n");
    // for (int i = 0; i < 10; i++) {
    //     printf("%e\n", val_csr[i]);
    // }
    // free(val_check);

    // float *result_check = (float*)malloc(10 * sizeof(float));
    // cudaMemcpy(result_check, d_result, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("\n[Check] First 10 values of d_result = alpha * d_val:\n");
    // for (int i = 0; i < 10; i++) {
    //     printf("GPU: %e   Expected: %e\n", result_check[i], val_csr[i] * alpha);
    // }
    // free(result_check);
    
    // Copy d_a from the device 
    // Get_CSR_From_Device(d_row_ptr, row_ptr, d_col_idx, col_idx, d_val, val_csr, MAX_NROWS, nnz);


    // // === Milestone 3: 向量初始化與 SpMV 呼叫 ===
    // float *h_vec = (float*)malloc(MAX_NROWS * sizeof(float));
    // for (int i = 0; i < MAX_NROWS; i++) h_vec[i] = 1.0f;  // 初值設為 1.0

    float *d_vec, *d_out;
    // cudaMalloc((void**)&d_vec, MAX_NROWS * sizeof(float));
    // cudaMalloc((void**)&d_out, MAX_NROWS * sizeof(float));
    // cudaMemcpy(d_vec, h_vec, MAX_NROWS * sizeof(float), cudaMemcpyHostToDevice);

    // Launch_SpMV_CSR(d_row_ptr, d_col_idx, d_val, d_vec, d_out, MAX_NROWS);

    // float *h_result = (float*)malloc(10 * sizeof(float));
    // cudaMemcpy(h_result, d_out, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("\n[SpMV] First 10 results:\n");
    // for (int i = 0; i < 10; i++) printf("%e\n", h_result[i]);

    // free(h_result); free(h_vec);



    // All of this commented code is driving me crazy.
    // So, here is a picture of a cat, written in ASCII.
    /*
	 /\_/\
	( o.o )
	 > ^ <
	- Prof. Smith
    */


    // // === Milestone 4: Dot Product 測試 ===
    float *d_vecA, *d_vecB, *d_dot;
    // cudaMalloc((void**)&d_vecA, MAX_NROWS * sizeof(float));
    // cudaMalloc((void**)&d_vecB, MAX_NROWS * sizeof(float));
    // cudaMalloc((void**)&d_dot, sizeof(float));

    // float *h_vecA = (float*)malloc(MAX_NROWS * sizeof(float));
    // float *h_vecB = (float*)malloc(MAX_NROWS * sizeof(float));
    // for (int i = 0; i < MAX_NROWS; i++) {
    //     h_vecA[i] = 1.0f;
    //     h_vecB[i] = 2.0f;
    // }
    // cudaMemcpy(d_vecA, h_vecA, MAX_NROWS * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_vecB, h_vecB, MAX_NROWS * sizeof(float), cudaMemcpyHostToDevice);

    // Launch_Dot_Product(d_vecA, d_vecB, d_dot, MAX_NROWS);

    // float h_dot_result;
    // cudaMemcpy(&h_dot_result, d_dot, sizeof(float), cudaMemcpyDeviceToHost);
    // printf("\n[Dot Product] Result = %f (Expected = %f)\n", h_dot_result, 2.0f * MAX_NROWS);

    // // Free dot product related memory
    // free(h_vecA); free(h_vecB);


    // === Milestone 5: CG Solver 測試 ===
    float *d_b, *d_x;
    cudaMalloc((void**)&d_b, MAX_NROWS * sizeof(float));
    cudaMalloc((void**)&d_x, MAX_NROWS * sizeof(float));

    // float *h_b = (float*)malloc(MAX_NROWS * sizeof(float));
    // for (int i = 0; i < MAX_NROWS; i++) {
    //     h_b[i] = 1.0f; // b 全設為 1
    // }
    
    // === 讀入 Fr_Final.txt 作為 b 向量 ===
    float *h_b = (float*)malloc(MAX_NROWS * sizeof(float));
    FILE *f_fr = fopen("Fr_Final.txt", "r");
    for (int i = 0; i < MAX_NROWS; i++) {
        if (fscanf(f_fr, "%f", &h_b[i]) != 1) {
            printf("Error reading Fr_Final.txt at index %d\n", i);
            return -1;
        }
    }
    printf("[Info] Loaded Fr_Final.txt into h_b (size = %d)\n", MAX_NROWS);
    // for (int i = 0; i < MAX_NROWS; i++) {
    //     fscanf(f_fr, "%f", &h_b[i]);
    // }
    fclose(f_fr);

    cudaMemcpy(d_b, h_b, MAX_NROWS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, MAX_NROWS * sizeof(float));     // x 初始為 0

    int max_iter = 10000;
    float tol = 1e-8;
    Launch_CG_Solver(d_row_ptr, d_col_idx, d_val, 
                    d_b, d_x, MAX_NROWS, max_iter, tol);

    // 抓前 10 筆解回來看
    float *h_x = (float*)malloc(MAX_NROWS * sizeof(float));
    cudaMemcpy(h_x, d_x, MAX_NROWS * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n[CG Solver] First 10 results of x:\n");
    for (int i = 0; i < 10; i++) {
        printf("x[%d] = %e\n", i, h_x[i]);
    }

    // Milestone 6: 計算最大變位
    float max_disp = 0.0f;
    for (int i = 0; i < MAX_NROWS / 2; i++) {
        float dx = h_x[2 * i];
        float dy = h_x[2 * i + 1];
        float disp = sqrtf(dx * dx + dy * dy);
        if (disp > max_disp)
            max_disp = disp;
    }
    printf("\n[Postprocess] Max displacement = %e\n", max_disp);

    FILE *fp_x = fopen("cg_solution.csv", "w");
    for (int i = 0; i < MAX_NROWS; i++) {
        fprintf(fp_x, "%e\n", h_x[i]);
    }
    fclose(fp_x);
    printf("Saved CG result to cg_solution.csv\n");
    
    free(h_b);free(h_x);


    // Free memory
    Free_CSR_Memory(&row, &col, &val, &row_ptr, &col_idx, &val_csr, &d_row_ptr, 
                    &d_col_idx, &d_val, &d_result, &d_vec, &d_out, 
                    &d_vecA, &d_vecB, &d_dot, &d_b, &d_x);
    return 0;
}

