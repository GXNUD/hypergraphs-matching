#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>
#include <stdio.h>

using namespace std;


__global__ void kernel_Sum(double *A, double *B, double *C, int N){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < N){
    C[i] = A[i] + B[i];
  }

}

int main(int argc, char *argv[])
{

    vector<double> a;
    double *ap = (double*)malloc((a.size())*sizeof(double));
    vector<double> b;
    double *bp = (double*)malloc((b.size())*sizeof(double));
    vector<double> c;
    double *cp = (double*)malloc((c.size())*sizeof(double));

    // declaración de variables cuda para la GPU
    double *d_Ap, *d_Bp, *d_Cp;
    
    a.push_back(999.25);
    a.push_back(888.50);
    a.push_back(777.25);

    b.push_back(999.25);
    b.push_back(888.50);
    b.push_back(777.25);
    c.push_back(0.0);
    c.push_back(0.0);
    c.push_back(0.0);

    int N = c.size();
    cout << N << '\n';

    cudaMalloc((void **)&d_Ap , N*sizeof(double));
    cudaMalloc((void **)&d_Bp , N*sizeof(double));
    cudaMalloc((void **)&d_Cp , N*sizeof(double));

    //int threadsPerBlock = 512;
    //int blocksPerGrid =  ceil(double(N)/double(threadsPerBlock));
    //
    for(int i = 0; i< a.size(); i++){
      ap[i]= a[i];
      bp[i]= b[i];
      cp[i]= c[i];
    }

    for(int i = 0; i < c.size(); i++)
    {
      cp[i]=ap[i]+bp[i];
      cout << cp[i] << endl;
    }
    cout << "--FINALIZA SECUENCIAL------" << endl;

    cudaMemcpy (d_Ap, ap , N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy (d_Bp, bp , N*sizeof(double),cudaMemcpyHostToDevice);

    kernel_Sum <<<1,1>>>(d_Ap, d_Bp, d_Cp, N);

    cudaMemcpy (cp, d_Cp , N*sizeof(double),cudaMemcpyHostToDevice);

    for(int i = 0; i< N; i++){
      cout << cp[i] << '\n';
    }

    cout << "---FINALIZA PARALELO-------" << endl;


    free(ap);
    free(cp);
    free(bp);
    cudaFree(d_Ap);
    cudaFree(d_Bp);
    cudaFree(d_Cp);

    return EXIT_SUCCESS;
}
