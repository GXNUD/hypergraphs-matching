#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <cuda.h>

using namespace std;


__global__ void kernel_Sum(int *A, int *B, int *C, int N){
  int idx = threadIdX.x +blockDim.x*blockIdx.x;
  if(idx < N){
    C[idx] = A[idx] + B[idx];
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
    int N = c.size();
    // declaración de variables cuda para la GPU
    double *d_Ap, *d_Bp, *d_Cp;

    cudaMalloc((void **)&dev_a , N*sizeof(double));
    cudaMalloc((void **)&dev_a , N*sizeof(double));
    cudaMalloc((void **)&dev_a , N*sizeof(double));


    a.push_back(999.25);
    a.push_back(888.50);
    a.push_back(777.25);

    b.push_back(999.25);
    b.push_back(888.50);
    b.push_back(777.25);

    c.push_back(0.0);
    c.push_back(0.0);
    c.push_back(0.0);

    int threadsPerBlock = 512;
    int blocksPerGrid =  ceil(double(N)/double(threadsPerBlock));

    cudaMemcpy (d_Ap, ap , N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy (d_Bp, bp , N*sizeof(double),cudaMemcpyHostToDevice);

    kernel_Sum<<<blocksPerGrid,threadsPerBlock>>>(d_Ap, d_Bp, d_Cp, N);

    cudaMemcpy (cp, d_Cp , N*sizeof(double),cudaMemcpyHostToDevice);

    for(int i = 0; i< N; i++){
      cout << cp[i] << '\n';
      cout << "----PARALELO------" << endl;
    }

    cudaFree(d_Ap);
    cudaFree(d_Bp);
    cudaFree(d_Cp);


    for(int i = 0; i< a.size(); i++){
      ap[i]= a[i];
      bp[i]= b[i];
      cp[i]= c[i];
      cout << ap[i] << '\n';
      cout << bp[i] << '\n';
      cout << cp[i] << '\n';
      cout << "----------" << endl;
    }

    cout << "----------" << endl;
    for(int i = 0; i < c.size(); i++)
    {
        c[i]=a[i]+b[i];
      cout << c[i] << endl;
    }
    cout << "----------" << endl;
    for(int i = 0; i < c.size(); i++)
    {
      cp[i]=ap[i]+bp[i];
      cout << cp[i] << endl;
    }
    cout << "----------" << endl;
    free(ap);
    free(cp);
    free(bp);

    return EXIT_SUCCESS;
}
