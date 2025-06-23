#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"
#include <math.h> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void Save_Results(float *h_T, int *h_Body, int NX, int NY, float DX, float DY) {
    FILE *fptr;
    const int N = NX*NY;
    fptr = fopen("results.txt", "w");
    // Write data to a file using tab delimiting
    for (int i = 0; i < N; i++) {
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;
        float cx = (xcell+0.5)*DX;
        float cy = (ycell+0.5)*DY;
        fprintf(fptr, "%g\t%g\t%d\t%g\n", cx, cy, h_Body[i], h_T[i]);
    }
    // Close the file
    fclose(fptr);
}


int main(int argc, char *argv[]) {

    float *h_T, *h_Tnew, *d_T, *d_Tnew;
    int *h_Body, *d_Body;
    int NX = 1070;
    int NY = 1070;
    int N = NX*NY;
    float L = 1.0;   // Length of region
    float H = 0.5;   // Height of region
    float W = 0.25;  // Hole size
    float DX = 0.18f / NX;
    float DY = 0.18f / NY;
    float DT = 0.0001;
    int NO_STEPS = 12000; // Damn I'm dumb stop me guys if I do that again!!!

    Set_GPU_Device(1);

    // Allocate memory on both device and host
    Allocate_Memory(&h_T, &h_Tnew, &h_Body, &d_T, &d_Tnew, &d_Body, N);

    // === 讀取 Brake.csv 初始化 h_Body 與 h_T ===
    FILE *fp = fopen("Brake.csv", "r");
    if (!fp) { printf("Unable to open Brake.csv\n"); exit(1); }

    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            int idx = i * NY + j;
            int val;
            fscanf(fp, "%d,", &val);
            h_Body[idx] = (val <= 200) ? 1 : 0;
            h_T[idx] = 300.0f;
        }
    }
    fclose(fp);

    // ========== 記錄所有 solid cell (body==1) 的 index ==========
    int* solid_indices = (int*)malloc(N * sizeof(int));
    int n_solid = 0;
    for (int i = 0; i < N; i++) {
        if (h_Body[i] == 1) solid_indices[n_solid++] = i;
    }


    // Take h_T and h_Body and store both on the device
    Send_To_Device(&h_T, &d_T, &h_Body, &d_Body, N);

    for (int step = 0; step < NO_STEPS; step++) {
    float time = step * DT;
    float omega = 2.0f * M_PI * (100.0f / 60.0f);
    int pad_count = 0;

    for (int idx = 0; idx < n_solid; idx++) {
        int i = solid_indices[idx] / NY;
        int j = solid_indices[idx] % NY;
        float cx = (i + 0.5f) * DX;
        float cy = (j + 0.5f) * DY;
        float pad_x = 0.09f + 0.08f * cosf(omega * time);
        float pad_y = 0.09f + 0.08f * sinf(omega * time);
        float dist = sqrtf((cx - pad_x)*(cx - pad_x) + (cy - pad_y)*(cy - pad_y));
        if (dist <= 0.005f)
            pad_count++;
    }

    Compute_GPU_Good(d_T, d_Tnew, d_Body, NX, NY, DX, DY, DT, step, pad_count);
    Update_Temperature_GPU(&d_T, &d_Tnew, N);
    
    }


    // Copy d_a from the device into h_b on the host
    Get_From_Device(&d_T, &h_T, N);

    // Save the result
    Save_Results(h_T, h_Body, NX, NY, DX, DY);

    // Free memory
    Free_Memory(&h_T, &h_Tnew, &h_Body, &d_T, &d_Tnew, &d_Body);
    free(solid_indices);

    return 0;
}