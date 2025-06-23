#include <stdio.h>
#include <stdlib.h>


#define a0 2.07e-5
#define a1 4.02e-6

void Set_GPU_Device(int device) {
    cudaSetDevice(device);
}

__device__ float Compute_Heat_Flux(float Temp, float Area){
    //Here we have a convective heat flux
    //Compute the convective heat flux on the device in a device function
    //The air temp is fixed
    float h = 0.1; //Estimate of h value
    return h*Area*(Temp - 300.0);
}

__global__  void Compute_New_Temperature_GPU_Good(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT, int step, int pad_count) {

    // Goal is to calculate h_Tnew
    int N = NX*NY;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        // We can use the index to compute the x, y cell.
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;

        float TC = d_T[i];
        float aC = (d_Body[i] == 0)*a0 + (d_Body[i] == 1)*a1;

        float TL, TR, TD, TU;
        float aL, aR, aD, aU;

        //Check right 
        if (xcell == (NX-1)){
            TR = TC;
            aR = aC;
        }else {
            if(d_Body[i+NY] == 0){
                TR = TC;                    
                aR = aC;
                }else{
            //We can check the cell to the right 
            TR = d_T[i+NY];
            aR = (d_Body[i+NY] == 0) ? a0 : a1;
            }
        }
        // Check left 
        if (xcell == 0){
            TL = TC;
            aL = aC;
        }else{
            if(d_Body[i-NY] == 0){
                TL = TC;
                aL = aC;
            }else{
            //We can check the cell to the right 
            TL = d_T[i-NY];
            aL = (d_Body[i-NY] == 0) ? a0 : a1;
            }
        }
        //Vertical direction now
        //Check up(U)
        if (ycell == (NY-1)) {
            TU = TC;                // 最上邊界，設自己溫度（絕熱）
            aU = aC;
        } else {
            if (d_Body[i+1] == 0) {
                TU = TC;            // 上面是空氣，設自己溫度（絕熱）
                aU = aC;
            } else {
                TU = d_T[i+1];      // 上面是金屬，取鄰近溫度
                aU = (d_Body[i+1] == 0) ? a0 : a1;
            }
        }
        // Check down (D)
        if (ycell == 0) {
            TD = TC;                // 最下邊界，設自己溫度（絕熱）
            aD = aC;
        } else {
            if (d_Body[i-1] == 0) {
                TD = TC;            // 下面是空氣，設自己溫度（絕熱）
                aD = aC;
            } else {
                TD = d_T[i-1];      // 下面是金屬，取鄰近溫度
                aD = (d_Body[i-1] == 0) ? a0 : a1;
            }
        }    

        // We are now ready to compute T_new
        d_Tnew[i] = d_T[i] + (2.0/( (1.0/aC) + (1.0/aR) ))*(DT/(DX*DX))*(TR - TC);
        d_Tnew[i] = d_Tnew[i] - (2.0/( (1.0/aC) + (1.0/aL) ))*(DT/(DX*DX))*(TC - TL);
        d_Tnew[i] = d_Tnew[i] + (2.0/( (1.0/aC) + (1.0/aU) ))*(DT/(DY*DY))*(TU - TC);
        d_Tnew[i] = d_Tnew[i] - (2.0/( (1.0/aC) + (1.0/aD) ))*(DT/(DY*DY))*(TC - TD);
    
        if (d_Body[i] == 1) {
        float time = step * DT;
        float omega = 2.0f * M_PI * (100.0f / 60.0f);
        float pad_x = 0.09f + 0.08f * cosf(omega * time);
        float pad_y = 0.09f + 0.08f * sinf(omega * time);

        int xcell = i / NY;
        int ycell = i % NY;
        float cx = (xcell + 0.5f) * DX;
        float cy = (ycell + 0.5f) * DY;
        float dist = sqrtf((cx - pad_x)*(cx - pad_x) + (cy - pad_y)*(cy - pad_y));

        // 正確加熱：pad_count>0 時才分配熱量
        if (dist <= 0.005f && pad_count > 0) {
            float mass = 8050.0f * DX * DY * 0.002f;
            float Q_per_cell = (12000.0f * DT) / pad_count;
            float dT_per_cell = Q_per_cell / (mass * 502.0f);
            d_Tnew[i] += dT_per_cell;
        }

        // 強制對流冷卻
        float R = sqrtf((cx - 0.09f)*(cx - 0.09f) + (cy - 0.09f)*(cy - 0.09f));
        float V = omega * R;
        float rho_air = 1.25f;
        float mu_air  = 1.81e-5f;
        float k_air   = 0.026f;
        if (R > 1e-4f) {
            float Re = (2.0f * rho_air * M_PI * R * V) / mu_air;
            float h = 0.664f * powf(Re, 0.5f) * powf(0.7f, 1.0f/3.0f) * k_air / (2.0f * M_PI * R);
            float mass = 8050.0f * DX * DY * 0.002f;
            float Q_loss = h * (d_Tnew[i] - 300.0f) * DX * DY * DT;
            float dT_cool = Q_loss / (mass * 502.0f);
            d_Tnew[i] -= dT_cool;
        }
    }
    }
}


// Write a simple wrapping function to call the crap GPU code
void Compute_GPU_Good(float *d_T, float *d_Tnew, int *d_Body, int NX, int NY, float DX, float DY, float DT, int step, int pad_count) {
    int N = NX*NY;
    int threadsperblock = 128;
    int blockspergrid = (N + threadsperblock - 1) / threadsperblock;
    Compute_New_Temperature_GPU_Good<<<blockspergrid,threadsperblock>>>(d_T, d_Tnew, d_Body, NX, NY, DX, DY, DT, step, pad_count);

}

void Allocate_Memory(float **h_T, float **h_Tnew, int **h_Body, 
                     float **d_T, float **d_Tnew, int **d_Body, int N) {
    cudaError_t Error;
    // Host memory
    *h_T = (float*)malloc(N*sizeof(float));
    *h_Tnew = (float*)malloc(N*sizeof(float)); 
    *h_Body = (int*)malloc(N*sizeof(int)); 
    // Device memory
    Error = cudaMalloc((void**)d_T, N*sizeof(float));
    printf("CUDA error (malloc d_T) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Tnew, N*sizeof(float));
    printf("CUDA error (malloc d_Tnew) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_Body, N*sizeof(int));
    printf("CUDA error (malloc d_Body) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory(float **h_T, float **h_Tnew, int **h_Body, 
                 float **d_T, float **d_Tnew, int **d_Body) {
    if (*h_T) free(*h_T);
    if (*h_Tnew) free(*h_Tnew);
    if (*h_Body) free(*h_Body);
    if (*d_T) cudaFree(*d_T);
    if (*d_Tnew) cudaFree(*d_Tnew);
    if (*d_Body) cudaFree(*d_Body);
}

void Send_To_Device(float **h_T, float **d_T, int **h_Body, int **d_Body, int N) {
    // Grab an error type
    cudaError_t Error;
    // Send T to the GPU
    Error = cudaMemcpy(*d_T, *h_T, N*sizeof(float), cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_T -> d_T) = %s\n", cudaGetErrorString(Error));
    // Send Body to the GPU
    Error = cudaMemcpy(*d_Body, *h_Body, N*sizeof(int), cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_Body -> d_Body) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device(float **d_T, float **h_T, int N) {
    // Grab a error type
    cudaError_t Error;
    // Send d_a to the host variable h_b
    Error = cudaMemcpy(*h_T, *d_T, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_T -> h_T) = %s\n", cudaGetErrorString(Error));
}

void Update_Temperature_GPU(float **d_T, float **d_Tnew, int N) {
    cudaError_t Error;
    // Send d_Tnew into d_T
    Error = cudaMemcpy(*d_T, *d_Tnew, N*sizeof(float), cudaMemcpyDeviceToDevice);
    //printf("CUDA error (memcpy d_Tnew -> d_T) = %s\n", cudaGetErrorString(Error));
}