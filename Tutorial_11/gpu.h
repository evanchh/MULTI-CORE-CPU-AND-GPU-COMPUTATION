/*
gpu.h
Declarations of functions used by gpu.cu
*/

void Set_GPU_Device(int device); 

void Allocate_CSR_Memory(int **d_row_ptr, int **d_col_idx, float **d_val, int NROWS, int NNZ);

// void Compute_GPU_Good(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT, int step, int pad_count);

// void Free_Memory(float **h_T, float **h_Tnew, int **h_Body, float **d_T, float **d_Tnew, int **d_Body);

void Send_CSR_To_Device(int **d_row_ptr, int *h_row_ptr, int **d_col_idx, int *h_col_idx, float **d_val, float *h_val, int NROWS, int NNZ);

void Get_CSR_From_Device(int *d_row_ptr, int *h_row_ptr, 
                         int *d_col_idx, int *h_col_idx,
                         float *d_val, float *h_val,
                         int NROWS, int NNZ);

// === Milestone 2: 向量 x 常數乘法 kernel 和呼叫函數 ===
void Launch_Vector_Multiply_Constant(float *d_y, float *d_x, float alpha, int N);

// === Milestone 3: Sparse Matrix-Vector Multiply ===
void Launch_SpMV_CSR(int *d_row_ptr, int *d_col_idx, float *d_val,
                     float *d_vec, float *d_out, int num_rows);

// === Milestone 4: Dot Product ===
void Launch_Dot_Product(float *d_a, float *d_b, float *d_result, float *d_tmp, int N);

// === Milestone 5: CG Method ===
void Launch_CG_Solver(int *d_row_ptr, int *d_col_idx, float *d_val,
                      float *d_b, float *d_x, int N, int max_iter, float tol);

void Launch_Vector_AXPY(float *d_y, float *d_x, float alpha, int N);


void Free_CSR_Memory(int **row, int **col, float **val, int **row_ptr, 
                     int **col_idx, float **val_csr, int **d_row_ptr, 
                     int **d_col_idx, float **d_val, float **d_result, 
                     float **d_vec, float **d_out, 
                     float **d_vecA, float **d_vecB, float **d_dot, 
                     float **d_b, float **d_x);