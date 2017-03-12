#include <stdio.h>
#include <stdlib.h>

#define SIZE 3


__global__ void d_hyperedges (int *edges1, int *edges2,
        float *kp1, float *kp2,
        float *desc1, float *desc2, double c1,
        double c2, double c3, double thresholding,
        int edges1Size, int edges2Size, int *matches){
    double sigma = 0.5;
    double sum = c1 + c2 + c3;
    c1 /= sum;
    c2 /= sum;
    c3 /= sum;

    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < edges1Size){
        int best_match_idx = -1;
        double max_similarity = -1E30;
        double s_ang = -1E30;
        double s_ratios = -1E30;
        double s_desc = -1E30;
        if(j < edges2Size){
            float e1_points[3][2], e2_points[3][2];
            __shared__ float des_1[64], des_2[64];
            e1_points[0][0] = kp1[edges1[i*SIZE+0]*2+0];
            e1_points[0][1] = kp1[edges1[i*SIZE+0]*2+1];
            e1_points[1][0] = kp1[edges1[i*SIZE+1]*2+0];
            e1_points[1][1] = kp1[edges1[i*SIZE+1]*2+1];
            e1_points[2][0] = kp1[edges1[i*SIZE+2]*2+0];
            e1_points[2][1] = kp1[edges1[i*SIZE+2]*2+1];
            e2_points[0][0] = kp2[edges2[j*SIZE+0]*2+0];
            e2_points[0][1] = kp2[edges2[j*SIZE+0]*2+1];
            e2_points[1][0] = kp2[edges2[j*SIZE+1]*2+0];
            e2_points[1][1] = kp2[edges2[j*SIZE+1]*2+1];
            e2_points[2][0] = kp2[edges2[j*SIZE+2]*2+0];
            e2_points[2][1] = kp2[edges2[j*SIZE+2]*2+1];
            if(i==5 && j==7)
                printf("Puntos: %f, %f, %f, %f \n",e1_points[0][0],e1_points[0][1],e1_points[1][0],e1_points[1][1]);

        }
    }
}

