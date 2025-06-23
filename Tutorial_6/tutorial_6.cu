#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void Allocate_Memory(float **h_a, float **h_b, float **d_a, int N) {
    size_t size = N*sizeof(float);
    cudaError_t Error;
    // Host memory
    *h_a = (float*)malloc(size); 
    *h_b = (float*)malloc(size); 
    // Device memory 
    Error = cudaMalloc((void**)d_a, size); 
    printf("CUDA error (malloc d_a) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_a, float **h_b, float **d_a) {
    if (*h_a) free(*h_a);
    if (*h_b) free(*h_b);
    if (*d_a) cudaFree(*d_a);
}

void Send_To_Device(float **h_a, float **d_a, int N) {
    // Size of data to send
    size_t size = N*sizeof(float);
    // Grab a error type
    cudaError_t Error;

    // Send A to the GPU
    Error = cudaMemcpy(*d_a, *h_a, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_a -> d_a) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device(float **d_a, float **h_b, int N) {
    // Size of data to send
    size_t size = N*sizeof(float);
    // Grab a error type
    cudaError_t Error;
    // Send d_a to the host variable h_bS
    Error = cudaMemcpy(*h_b, *d_a, size, cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_a -> h_b) = %s\n", cudaGetErrorString(Error));
}

int main() {
    // 宣告變數
    float *h_T, *h_b, *d_T, *d_Tnew;
    int *h_Body, *d_Body;
    int N = 10000;

    // 使用封裝好的函式分配 h_T, h_b, d_T
    Allocate_Memory(&h_T, &h_b, &d_T, N);

    // 分配 d_Tnew
    cudaError_t Error;
    Error = cudaMalloc((void**)&d_Tnew, N * sizeof(float));
    if (Error != cudaSuccess) {
        printf("CUDA error (malloc d_Tnew) = %s\n", cudaGetErrorString(Error));
        return 1;
    }

    // 分配 h_Body 和 d_Body
    h_Body = (int*)malloc(N * sizeof(int));
    if (h_Body == NULL) {
        printf("malloc failed for h_Body\n");
        return 1;
    }

    Error = cudaMalloc((void**)&d_Body, N * sizeof(int));
    if (Error != cudaSuccess) {
        printf("CUDA error (malloc d_Body) = %s\n", cudaGetErrorString(Error));
        return 1;
    }

    // 初始化 h_T = 1.0f，h_Body = 0
    for (int i = 0; i < N; i++) {
        h_T[i] = 1.0f;
        h_Body[i] = 0;
    }

    // 儲存 h_T 到檔案
    FILE *fp = fopen("output_h_T.txt", "w");
    if (fp == NULL) {
        printf("Failed to open file for writing.\n");
        return 1;
    }
    for (int i = 0; i < N; i++) {
        fprintf(fp, "%f\n", h_T[i]);
    }
    fclose(fp);

    // 釋放所有記憶體
    Free_Memory(&h_T, &h_b, &d_T);
    free(h_Body);
    cudaFree(d_Tnew);
    cudaFree(d_Body);

    printf("Program completed successfully.\n");
    return 0;
}