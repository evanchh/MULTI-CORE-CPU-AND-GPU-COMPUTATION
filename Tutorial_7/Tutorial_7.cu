#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NX 100
#define NY 100
#define Lx 1.0f
#define Ly 0.5f

void Allocate_Memory(float **h_a, float **h_b, float **d_a, int size) {
    size_t bytes = size * sizeof(float);
    cudaError_t Error;

    *h_a = (float*)malloc(bytes);
    *h_b = (float*)malloc(bytes);
    Error = cudaMalloc((void**)d_a, bytes);
    printf("CUDA error (malloc d_a) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_a, float **h_b, float **d_a) {
    if (*h_a) free(*h_a);
    if (*h_b) free(*h_b);
    if (*d_a) cudaFree(*d_a);
}

void Send_To_Device(float **h_a, float **d_a, int size) {
    size_t bytes = size * sizeof(float);
    cudaError_t Error = cudaMemcpy(*d_a, *h_a, bytes, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_a -> d_a) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device(float **d_a, float **h_b, int size) {
    size_t bytes = size * sizeof(float);
    cudaError_t Error = cudaMemcpy(*h_b, *d_a, bytes, cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_a -> h_b) = %s\n", cudaGetErrorString(Error));
}

int main() {
    float *h_T, *h_b, *d_T, *d_Tnew;
    int *h_Body, *d_Body;
    cudaError_t Error;
    int total_size = NX * NY;

    float DX = Lx / NX;
    float DY = Ly / NY;

    // 分配溫度記憶體
    Allocate_Memory(&h_T, &h_b, &d_T, total_size);

    // 分配 d_Tnew
    Error = cudaMalloc((void**)&d_Tnew, total_size * sizeof(float));
    if (Error != cudaSuccess) {
        printf("CUDA error (malloc d_Tnew) = %s\n", cudaGetErrorString(Error));
        return 1;
    }

    // 分配 h_Body / d_Body
    h_Body = (int*)malloc(total_size * sizeof(int));
    Error = cudaMalloc((void**)&d_Body, total_size * sizeof(int));
    if (!h_Body || Error != cudaSuccess) {
        printf("Memory allocation failed for h_Body or d_Body.\n");
        return 1;
    }

    // 初始化 h_Body = 1
    for (int iy = 0; iy < NY; iy++) {
        for (int ix = 0; ix < NX; ix++) {
            h_Body[iy * NX + ix] = 1;
        }
    }

    // 左洞：x = 0.125~0.375, y = 0.05~0.3
    for (int iy = 10; iy < 60; iy++) {      // 0.05 / 0.005 = 10, 0.3 / 0.005 = 60
        for (int ix = 12; ix < 37; ix++) {  // 0.125 / 0.01 = 12, 0.375 / 0.01 = 37
            h_Body[iy * NX + ix] = 0;
        }
    }

    // 右洞：x = 0.625~0.875, y = 0.2~0.45
    for (int iy = 40; iy < 90; iy++) {      // 0.2 / 0.005 = 40, 0.45 / 0.005 = 90
        for (int ix = 62; ix < 87; ix++) {  // 0.625 / 0.01 = 62, 0.875 / 0.01 = 87
            h_Body[iy * NX + ix] = 0;
        }
    }

    // 初始化 h_T = 1.0f
    for (int i = 0; i < total_size; i++) {
        h_T[i] = 1.0f;
    }

    // 拷貝資料到 GPU
    Send_To_Device(&h_T, &d_T, total_size);
    Send_To_Device((float**)&h_Body, (float**)&d_Body, total_size);

    // 複製 d_T 到 d_Tnew
    Error = cudaMemcpy(d_Tnew, d_T, total_size * sizeof(float), cudaMemcpyDeviceToDevice);
    printf("CUDA error (copy d_T -> d_Tnew) = %s\n", cudaGetErrorString(Error));

    // CPU上將 h_T 清為 0
    for (int i = 0; i < total_size; i++) {
        h_T[i] = 0.0f;
    }

    // 將 d_Tnew 拷貝回 h_T
    Get_From_Device(&d_Tnew, &h_T, total_size);

    // 儲存結果至檔案
    FILE *fp = fopen("results.txt", "w");
    if (!fp) {
        printf("Failed to open results.txt\n");
        return 1;
    }

    for (int iy = 0; iy < NY; iy++) {
        for (int ix = 0; ix < NX; ix++) {
            int idx = iy * NX + ix;
            float x = ix * DX;
            float y = iy * DY;
            fprintf(fp, "%f %f %d %f\n", x, y, h_Body[idx], h_T[idx]);
        }
    }
    fclose(fp);

    // 清理記憶體
    Free_Memory(&h_T, &h_b, &d_T);
    free(h_Body);
    cudaFree(d_Tnew);
    cudaFree(d_Body);

    printf("Program completed successfully.\n");
    return 0;
}
