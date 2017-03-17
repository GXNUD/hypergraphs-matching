#include <stdio.h>
#include <stdlib.h>

#define SIGMA 0.5


__global__ void d_hyperedges (int *edges1, int *edges2,
        float *kp1, float *kp2,
        float *desc1, float *desc2, double c1,
        double c2, double c3, double thresholding,
        int edges1Size, int edges2Size, float *matches){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float *e1_points,*e2_points;
    e1_points = (float*) malloc(6*sizeof(float));
    e2_points = (float*) malloc(6*sizeof(float));

    if (i < edges1Size){
        for (int j = 0; j < edges2Size; j++) {
            e1_points[0*2+0] = kp1[(edges1[i*3+0])*2+0];
            e1_points[0*2+1] = kp1[(edges1[i*3+0])*2+1];
            e1_points[1*2+0] = kp1[(edges1[i*3+1])*2+0];
            e1_points[1*2+1] = kp1[(edges1[i*3+1])*2+1];
            e1_points[2*2+0] = kp1[(edges1[i*3+2])*2+0];
            e1_points[2*2+1] = kp1[(edges1[i*3+2])*2+1];
            e2_points[0*2+0] = kp2[(edges2[j*3+0])*2+0];
            e2_points[0*2+1] = kp2[(edges2[j*3+0])*2+1];
            e2_points[1*2+0] = kp2[(edges2[j*3+1])*2+0];
            e2_points[1*2+1] = kp2[(edges2[j*3+1])*2+1];
            e2_points[2*2+0] = kp2[(edges2[j*3+2])*2+0];
            e2_points[2*2+1] = kp2[(edges2[j*3+2])*2+1];
            matches[i*edges2Size+j] = e2_points[1];
        }
    }

    free(e1_points);free(e2_points);

}
