#include "d_similarity.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ int d_hyperedges (int *edges1, int *edges2,
        d_key_points *kp1, d_key_points *kp,
        d_Mat *desc1, d_Mat *desc2, double c1,
        double c2, double c3, double thresholding, int *matches){
    double sigma = 0.5;
    double sum = c1 + c2 + c3;
    c1 /= sum;
    c2 /= sum;
    c3 /= sum;

    int i = blockIdx.y*blocDim.y + threadIdx.y;
    int j = blockIdx.y*blocDim.y + threadIdx.y;

    if (i < edges1Size)

}

