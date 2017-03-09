#include "d_similarity.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define SIZE 3


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

    if (i < edges1Size){
        int best_match_idx = -1;
        double max_similarity = -1E30;
        double s_ang = -1E30;
        double s_ratios = -1E30;
        if(j < edges2Size){
            d_Point2f e1_points[3], e2_points[3];
            d_Mat des_1[3], des_2[3];
            #pragma unroll
            for (int k = 0; k < SIZE; k++){
                e1_points[k].x = kp1[edges1[i*WIDTH_EDGES1+k]].x;
                e1_points[k].y = kp1[edges1[i*WIDTH_EDGES1+k]].y;
                e2_points[k].x = kp2[edges2[i*WIDTH_EDGES2+k]].x;
                e2_points[k].y = kp2[edges2[i*WIDTH_EDGES2+k]].y;
                for(int h = 0; h < desc1Size; h++)
                    des_1[k] = desc1[edges1[i*WIDTH_EDGES1+h]];
                for(h = 0; h < desc2Size; h++)
                    des_2[k] = desc2[edges2[i*WIDTH_EDGES2+h]];
            }
            double sim_angles = d_sim::angles(e1_points, e2_points, sigma);
            double sim_ratios = d_sim::ratios(e1_points, e2_points, sigma);
            double sim_desc = sim::descriptors(des_1, des_2, sigma);

        }
    }

}

