/*
gpu.h
Declarations of functions used by gpu.cu
*/

#define a0 2.07e-5
#define a1 4.02e-6

void Set_GPU_Device(int device); 

void Allocate_Memory(float **h_T, float **h_Tnew, int **h_Body, float **d_T, float **d_Tnew, int **d_Body, int N);

// void Compute_GPU_Crap(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT);

void Compute_GPU_Good(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT, int step, int pad_count);

void Free_Memory(float **h_T, float **h_Tnew, int **h_Body, float **d_T, float **d_Tnew, int **d_Body);

void Send_To_Device(float **h_T, float **d_T, int **h_Body, int **d_Body, int N);

void Get_From_Device(float **d_T, float **h_T, int N);

void Update_Temperature_GPU(float **d_T, float **d_Tnew, int N);