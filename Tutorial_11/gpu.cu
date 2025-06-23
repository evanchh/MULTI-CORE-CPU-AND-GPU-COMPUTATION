#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void Set_GPU_Device(int device) {
    cudaSetDevice(device);
}

void Allocate_CSR_Memory(int **d_row_ptr, int **d_col_idx, float **d_val, int NROWS, int NNZ) {
    cudaError_t Error;
    // Device memory
    Error = cudaMalloc((void**)d_row_ptr, (NROWS + 1) * sizeof(int));
    printf("CUDA error (malloc d_row_ptr) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_col_idx, NNZ * sizeof(int));
    printf("CUDA error (malloc d_col_idx) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_val, NNZ * sizeof(float));
    printf("CUDA error (malloc d_val) = %s\n", cudaGetErrorString(Error));
}

void Free_CSR_Memory(int **row, int **col, float **val,
                     int **row_ptr, int **col_idx, float **val_csr,
                     int **d_row_ptr, int **d_col_idx, float **d_val, 
                     float **d_result, float **d_vec, float **d_out, 
                     float **d_vecA, float **d_vecB, float **d_dot, 
                     float **d_b, float **d_x) {
    if (*row) free(*row);
    if (*col) free(*col);
    if (*val) free(*val);
    if (*row_ptr) free(*row_ptr);
    if (*col_idx) free(*col_idx);
    if (*val_csr) free(*val_csr);
    if (*d_row_ptr) cudaFree(*d_row_ptr);
    if (*d_col_idx) cudaFree(*d_col_idx);
    if (*d_val) cudaFree(*d_val);
    if (*d_result) cudaFree(*d_result);
    if (*d_vec) cudaFree(*d_vec);
    if (*d_out) cudaFree(*d_out);
    if (*d_vecA) cudaFree(*d_vecA);
    if (*d_vecB) cudaFree(*d_vecB);
    if (*d_dot) cudaFree(*d_dot);
    if (*d_b) cudaFree(*d_b);
    if (*d_x) cudaFree(*d_x);
}

// === COO -> CSR 上傳到 GPU ===
void Send_CSR_To_Device(int **d_row_ptr, int *h_row_ptr,
                        int **d_col_idx, int *h_col_idx,
                        float **d_val, float *h_val,
                        int NROWS, int NNZ) {
    // Grab an error type
    cudaError_t Error;

    Error = cudaMemcpy(*d_row_ptr, h_row_ptr, (NROWS + 1) * sizeof(int), cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_row_ptr -> d_row_ptr) = %s\n", cudaGetErrorString(Error));

    Error = cudaMemcpy(*d_col_idx, h_col_idx, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_col_idx -> d_col_idx) = %s\n", cudaGetErrorString(Error));

    Error = cudaMemcpy(*d_val, h_val, NNZ * sizeof(float), cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_val -> d_val) = %s\n", cudaGetErrorString(Error));
}

void Get_CSR_From_Device(int *d_row_ptr, int *h_row_ptr, 
                         int *d_col_idx, int *h_col_idx,
                         float *d_val, float *h_val,
                         int NROWS, int NNZ) {
    cudaError_t Error;

    Error = cudaMemcpy(h_row_ptr, d_row_ptr, (NROWS + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_row_ptr -> h_row_ptr) = %s\n", cudaGetErrorString(Error));

    Error = cudaMemcpy(h_col_idx, d_col_idx, NNZ * sizeof(int), cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_col_idx -> h_col_idx) = %s\n", cudaGetErrorString(Error));

    Error = cudaMemcpy(h_val, d_val, NNZ * sizeof(float), cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_val -> h_val) = %s\n", cudaGetErrorString(Error));
}

// === Milestone 2: 向量 x 常數乘法 kernel 和呼叫函數 ===
__global__ void Vector_Multiply_Constant(float *out, float *in, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = alpha * in[idx];
    }
}

void Launch_Vector_Multiply_Constant(float *d_out, float *d_in, float alpha, int N) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    Vector_Multiply_Constant<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, alpha, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); // optional for debug
}

// === Milestone 3: 矩陣 × 向量 (current implementation target) ===
__global__ void SpMV_CSR_Kernel(int *row_ptr, int *col_idx, float *val,
                                 float *vec, float *out, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int i = start; i < end; i++) {
            dot += val[i] * vec[col_idx[i]];
        }
        out[row] = dot;
    }
}

void Launch_SpMV_CSR(int *d_row_ptr, int *d_col_idx, float *d_val,
                     float *d_vec, float *d_out, int num_rows) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    SpMV_CSR_Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_row_ptr, d_col_idx, d_val, d_vec, d_out, num_rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error (SpMV): %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// === Milestone 4: 向量內積 (Dot Product) ===
__global__ void DotProduct_Kernel(float *a, float *b, float *result, float *z, int N) {
    __shared__ float cache[128];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0f;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;

    __syncthreads();
    // Reduction
    // int i = blockDim.x / 2;
    // while (i != 0) {
    //     if (cacheIndex < i)
    //         cache[cacheIndex] += cache[cacheIndex + i];
    //     __syncthreads();
    //     i /= 2;
    // }
    // if (cacheIndex == 0)
    //     atomicAdd(result, cache[0]); //I explained why atomic add was bad. Why did you decide to use atomic add here?

    for (int i = blockDim.x / 2; i > 0; i = i /2 ){
        if(cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
    }
    if (cacheIndex == 0) 
        z[blockIdx.x] = cache[0];
}

__global__ void FinalReduction(float *z, float *result, int size) {
    __shared__ float cache[128];
    int tid = threadIdx.x;

    if (tid < size)
        cache[tid] = z[tid];
    else
        cache[tid] = 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        result[0] = cache[0];
}


void Launch_Dot_Product(float *d_a, float *d_b, float *d_result, float *d_tmp, int N) {
    cudaMemset(d_result, 0, sizeof(float));
    cudaMalloc(&d_tmp, 128 * sizeof(float));  // 分配 block 數量大小的暫存空間
    DotProduct_Kernel<<<128, 128>>>(d_a, d_b, d_result, d_tmp, N);
    FinalReduction<<<1, 128>>>(d_tmp, d_result, 128);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error (Dot Product): %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

__global__ void Vector_AXPY(float *y, float *x, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += alpha * x[idx];
}

void Launch_Vector_AXPY(float *d_y, float *d_x, float alpha, int N) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    Vector_AXPY<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_x, alpha, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error (AXPY): %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); // optional for debug
}

// === Milestone 5: 共軛梯度法 CG Solver ===
void Launch_CG_Solver(int *d_row_ptr, int *d_col_idx, float *d_val,
                      float *d_b, float *d_x, int N, int max_iter, float tol) {
    float *d_r, *d_p, *d_Ap, *d_rr, *d_pAp, *d_tmp;
    cudaMalloc(&d_r, N * sizeof(float));
    cudaMalloc(&d_p, N * sizeof(float));
    cudaMalloc(&d_Ap, N * sizeof(float));
    cudaMalloc(&d_rr, sizeof(float));
    cudaMalloc(&d_pAp, sizeof(float));
    cudaMalloc(&d_tmp, 256 * sizeof(float));

    // r = b - A*x
    Launch_SpMV_CSR(d_row_ptr, d_col_idx, d_val, d_x, d_Ap, N);          // Ap = A*x
    cudaMemcpy(d_r, d_b, N * sizeof(float), cudaMemcpyDeviceToDevice);   // r = b
    // Launch_Vector_Multiply_Constant(d_Ap, d_Ap, -1.0f, N);               // Ap = -Ap
    Launch_Vector_AXPY(d_r, d_Ap, -1.0f, N);                             // r = r + (-Ap)
    cudaMemcpy(d_p, d_r, N * sizeof(float), cudaMemcpyDeviceToDevice);   // p = r

    float alpha, beta, r_old, r_new;
    Launch_Dot_Product(d_r, d_r, d_rr, d_tmp, N);
    cudaMemcpy(&r_old, d_rr, sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < max_iter && r_old > tol * tol; ++i) {
        Launch_SpMV_CSR(d_row_ptr, d_col_idx, d_val, d_p, d_Ap, N);
        Launch_Dot_Product(d_p, d_Ap, d_pAp, d_tmp, N);
        float pAp;
        cudaMemcpy(&pAp, d_pAp, sizeof(float), cudaMemcpyDeviceToHost);

        alpha = r_old / pAp;

        Launch_Vector_AXPY(d_x, d_p, alpha, N);                  // x += alpha * p
        Launch_Vector_AXPY(d_r, d_Ap, -alpha, N);                 // r -= alpha * Ap

        Launch_Dot_Product(d_r, d_r, d_rr, d_tmp, N);                   // r_new = dot(r, r)
        cudaMemcpy(&r_new, d_rr, sizeof(float), cudaMemcpyDeviceToHost);

        if (r_new < tol * tol) break; // 收斂條件提前結束

        beta = r_new / r_old;
        Launch_Vector_Multiply_Constant(d_p, d_p, beta, N);  // p = β*p
        Launch_Vector_AXPY(d_p, d_r, 1.0f, N);               // p = β*p + r

        r_old = r_new;
    }

    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
    cudaFree(d_rr); cudaFree(d_pAp); cudaFree(d_tmp);
}