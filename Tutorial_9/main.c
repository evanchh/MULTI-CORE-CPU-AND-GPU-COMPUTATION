#include <stdio.h>
#include "gpu.h"

void Save_Results(float *h_T, float *h_Tnew, int *h_Body, int NX, int NY, float DX, float DY){
    FILE *fptr;
    const int N = NX*NY;
    fptr = fopen("results.txt", "w");
    for (int i = 0; i < N ; i++){
        int xcell = (int)i/NY;
        int ycell = i -xcell*NY;
        float cx = (xcell+0.5)*DX;
        float cy = (ycell+0.5)*DY;
        fprintf(fptr, "%g\t%g\t%d\t%g\n", cx, cy, h_Body[i], h_T[i]);
    }
    // Close the file
    fclose(fptr);
}

void Compute_New_Temperature(float *h_T, float *h_Tnew, int *h_Body, int NX, int NY, float DX, float DY, float DT){

    //Goal is to calculate h_Tnew 
    int N = NX*NY;

    for(int i = 0; i < N; i++){
        // we can use the index to compute the x, y cell.
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;

        float TC = h_T[i];
        float aC = (h_Body[i] == 0)*a0 + (h_Body[i] == 1)*a1 ;
        
        float TL, TR, TD, TU;
        float aL, aR, aD, aU;

        //Check right 
        if (xcell == (NX-1)){
            //We are on the right edge, its damn hot here.
            TR = 1000.0;
            aR = (h_Body[i] == 0)*a0 + (h_Body[i] == 1)*a1;
        }else {
            //We can check the cell to the right 
            TR = h_T[i+NY];
            aR = (h_Body[i+NY] == 0)*a0 + (h_Body[i+NY] == 1)*a1;
        }
        // Check left 
        if (xcell == 0){
            //We are ont the top left edge, its cool here.
            TL = 300.0;
            aL = (h_Body[i] == 0)*a0 + (h_Body[i] == 1)*a1;
        }else{
            //We can check the cell to the right 
            TL = h_T[i-NY];
            aL = (h_Body[i-NY] == 0)*a0 + (h_Body[i-NY] == 1)*a1;
        }

        //Vertical direction now
        //Check up(U)
        if (ycell == (NY-1)){
            //We are on the top edge, no heat flow.
            TU = h_T[i];
            aU = (h_Body[i] == 0)*a0 + (h_Body[i] == 1)*a1;
        }else {
            //We can check the cell above 
            TU = h_T[i+1];
            aU = (h_Body[i+1] == 0)*a0 + (h_Body[i+1] == 1)*a1;
        }
        // Check down (D) 
        if (ycell == 0){
            //We are ont the bottom edge, its also insulated 
            TD = h_T[i];
            aD = (h_Body[i] == 0)*a0 + (h_Body[i] == 1)*a1;
        }else{
            //We can check the cell to bottom 
            TD = h_T[i-1];
            aD = (h_Body[i-1] == 0)*a0 + (h_Body[i-1] == 1)*a1;
        }

        h_Tnew[i] = h_T[i] + (2.0/( (1.0/aC) + (1.0/aR) ))*(DT/(DX*DX))*(TR - TC);
        h_Tnew[i] = h_Tnew[i] - (2.0/( (1.0/aC) + (1.0/aL) ))*(DT/(DX*DX))*(TC - TL);
        h_Tnew[i] = h_Tnew[i] + (2.0/( (1.0/aC) + (1.0/aU) ))*(DT/(DY*DY))*(TU - TC);
        h_Tnew[i] = h_Tnew[i] - (2.0/( (1.0/aC) + (1.0/aD) ))*(DT/(DY*DY))*(TC - TD);
    }
}


int main(int argc, char *argv[]) {
// int cpu_demo(int argc, char *argv[]){
    // 宣告變數

    float *h_T, *h_Tnew, *d_T, *d_Tnew;
    int *h_Body, *d_Body;
    int NX = 200;
    int NY = 200;
    int N = NX * NY;
    float L = 1.0;      //Length of region 
    float H = 0.5;      //Height of region
    float W = 0.25;     //Hole size
    float DX = L / NX;   
    float DY = H / NY;
    float DT = 0.01;
    float NO_STEPS = 800000;
    size_t size = N*sizeof(float);

    Set_GPU(2);

    // 使用封裝好的函式分配 h_T, h_b, d_T
    Allocate_Memory(&h_T, &h_Tnew , &h_Body, &d_T, &d_Tnew, &d_Body, N);

    // 初始化 h_Body = 0
    for (int i = 0; i < N; i++) {
        // We can use the index to compute the x,y cell.
        int xcell = (int)i/NY;
        int ycell = i - xcell*NY;
        float cx = (xcell+0.5)*DX;
        float cy = (ycell+0.5)*DY;
        
        h_T[i] = 300.0; //Set the initial temperature everywhere to 300
        
        //Set the body 
        h_Body[i] = 0;
        //First hole 
        if ( (cx > 0.125) && (cx < (0.125 + W)) && (cy > 0.05) && (cy < 0.05 + W)) {
            h_Body[i] = 1;
        }
        //Second hole
        if ( (cx >(L-0.125-W)) && (cx < (L-0.125)) && (cy > (H-0.05-W)) && (cy < (H-0.05))) {
            h_Body[i] = 1;
        } 
    }

    Send_To_Device(&h_T, &d_T, &h_Body, &d_Body, N);
    
    // Take time steps 
    for (int step = 0; step < NO_STEPS; step++){
        // printf("Computing step %d of %.0f\n", step, NO_STEPS);
        // Calculate the new T (h_Tnew)
        // Compute_New_Temperature(h_T, h_Tnew, h_Body, NX, NY, DX, DY, DT);
        Compute_GPU_Good(d_T, d_Tnew, d_Body, NX,  NY, DX, DY, DT);

        // Updata the temperature 
        // for (int i = 0; i < N ; i++){
        //     h_T[i] = h_Tnew[i];
        // }
        Update_Temperature_GPU(&d_T, &d_Tnew, N);
    }

    // Send_To_Device(&h_T, &d_T, &h_Body, &d_Body, N);
    
    // 從 d_Tnew 抓回 h_T
    Get_From_Device(&d_T, &h_T, N);

    // Save the result
    Save_Results(h_T, h_Tnew, h_Body, NX, NY, DX, DY);

    // 釋放所有記憶體
    Free_Memory(&h_T, &h_Tnew, &h_Body, &d_T, &d_Tnew, &d_Body);

    printf("Tutorial_8.cu completed.\n");
    return 0;
}