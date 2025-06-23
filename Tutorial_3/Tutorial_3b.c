#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int N = 500;
float b[500];
float a[500];
const int  no_steps = 500;

const float phi = 0.25;



int main() {
        //init()
        for (int i = 0; i<N; i++){
                if (i< (0.5 * N)){
                        a[i] = 0.0;
                } else{
                        a[i] = 1.0;
                }

        }

        omp_set_num_threads(8);
        #pragma omp parallel

        for (int step = 0 ; step < no_steps; step++){

                printf ("Step %d of %d\n" , step, no_steps);
                
                //Calc()
                #pragma omp for
                for (int i = 0; i < N; i++){
                        float left ;
                        float right ;

                        if (i == 0) {
                                left = 0.0;
                        } else{
                                left = a[i-1];
                        }

                        if (i == (N-1)){
                                right = 1.0;  
                        } else{
                                right = a[i+1];
                        }

                        b[i] = a[i] + phi * (left + right - 2.0 * a[i]);
                }
                #pragma omp for
                for (int i = 0; i < N ; i++){
                        a[i] = b[i];
                }

        }

        //save
        {
                FILE * fp = fopen("result2.txt","w");
                if (fp == NULL) {
                        printf("Error opening file. \n");
                        return 1;
                }
                for (int i = 0; i<N; i++){
                        printf("a[%d] = %g\n",i, a[i]);
                        fprintf(fp, "%g\n", a[i]);
                        
                }
                fclose(fp);

        }

        printf("Data saved to result2.txt\n");
        return 0;

}
