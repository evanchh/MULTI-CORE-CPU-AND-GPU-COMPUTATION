#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 3        // 問題中只有 3 個未知數
#define ITER 10



//定義一個Matrix_Mutiply
void Matrix_Multiply(double *A, double *x, double *result, int row, int col){
	int index;
	for (int i = 0; i < row; i++) {
	   result[i] = 0.0;
	   for(int j = 0; j < col; j++) {
		index = i*col + j;
		result[i] += A[index]*x[j];
	   }
	}
}


// 主程式: 以 3x3 的 A 和 b=[1,0,0] 為例, 做 10 次迭代
int main()
{
    omp_set_num_threads(1); // 強制 OpenMP 使用單執行緒
    
    double A[9] = {2.0,-1.0,0.0, -1.0,2.0,-1.0, 0.0,-1.0,2.0 };
    double B[3] = {1.0, 0.0, 0.0};

    // 3) 解向量 x, 以及 R, P, AP, 用來做 CG 的運算
    double x[N], R[N], P[N], AP[N];

    //--------------------------------------------------------------------------
    // (a) 初始猜測 x=0
    //     R = B - A*x (但 x=0 => R=B)
    //     P = R
    //--------------------------------------------------------------------------
    for(int i = 0; i < N; i++){
        x[i] = 0.0;
    }

    double AX[N];
    Matrix_Multiply(A, x, AX, N, N);

    for (int i = 0; i< N; i++){
         R[i] = B[i] -AX[i];
         P[i] = R[i];
    }


    //--------------------------------------------------------------------------
    // (b) 進入 parallel 區域，只進一次
    //     在區域內，我們對每個迴圈加 #pragma omp for
    //--------------------------------------------------------------------------
    double PTAP = 0.0 ;
    double RTR = 0.0 ;
    double RTR_new = 0.0 ;

    #pragma omp parallel
    {

        for(int k = 0; k < ITER; k++){

	    //double PTAP = 0.0;
	    //double RTR = 0.0;
	    //double RTR_new = 0.0;

	        // (1) AP = A * P
            #pragma omp for
            for(int i = 0; i < N; i++){
                double sum = 0.0;
                for(int j = 0; j < N; j++){
                    sum += A[i*N + j] * P[j];
                }
                AP[i] = sum;
            }


            // (2) 計算 P^T AP 與 R^T R (用 reduction 來安全加總)
            #pragma omp for reduction(+:PTAP)
            for(int i = 0; i < N; i++){
                PTAP += P[i] * AP[i];
            }


            #pragma omp for reduction(+:RTR)
            for(int i = 0; i < N; i++){
                RTR += R[i] * R[i];
            }

            // (3) 由於 alpha 與 RTR_Record[k] 只需計算一次
            double alpha;
            #pragma omp single
            {
                alpha = RTR / PTAP;          // alpha = (R^T R)/(P^T AP)
            }

            // (4) x = x + alpha * P
            #pragma omp for
            for(int i = 0; i < N; i++){
                x[i] += alpha * P[i];
            }

            // (5) R = R - alpha * AP
            #pragma omp for
            for(int i = 0; i < N; i++){
                R[i] -= alpha * AP[i];
            }

            // (6) 計算新的 R^T R
            #pragma omp for reduction(+:RTR_new)
            for(int i = 0; i < N; i++){
                RTR_new += R[i] * R[i];
            }

            // (7) beta = (R^T R)_new / (R^T R)_old
            //     更新 P = R + beta * P
            double beta;
            #pragma omp single
            {
                beta = RTR_new / RTR;
            }

            #pragma omp for
            for(int i = 0; i < N; i++){
                P[i] = R[i] + beta * P[i];
            }

        } // end for k
    } // end #pragma omp parallel

    //--------------------------------------------------------------------------
    // (c) 輸出結果
    //--------------------------------------------------------------------------
    printf("After %d iterations:\n", ITER);
    printf("x = [ ");
    for(int i = 0; i < N; i++){
        printf("%f ", x[i]);
    }
    printf("]\n\n");

    return 0;
}

