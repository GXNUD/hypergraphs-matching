#include <stdio.h>
#include <stdlib.h>

#define SIGMA 0.5

__device__ float d_angle(float2 *p, int i){
    float2 a;
    float2 b;
    a.x = p[i].x - p[(i+1)%3].x;
    a.y = p[i].y - p[(i+1)%3].y;
    b.x = p[i].x - p[(i+2)%3].x;
    b.y = p[i].y - p[(i+2)%3].y;

    float dot = a.x*b.x+a.y*b.y;
    float angle = acos(dot/sqrt((a.x*a.x)+(a.y*a.y))
            /sqrt((b.x*b.x)+(b.y*b.y)));
    return sin(angle);
}

__device__ float d_sim_angles(float2 *p, float2 *q, int2 *idx){
    int i1 = idx[0].x;
    int j1 = idx[0].y;
    int i2 = idx[1].x;
    int j2 = idx[1].y;
    int i3 = idx[2].x;
    int j3 = idx[2].y;
    float mean = ((sin(d_angle(p,i1))-sin(d_angle(q,j1))) +
        (sin(d_angle(p,i2))-sin(d_angle(q,j2))) +
        (sin(d_angle(p,i3))-sin(d_angle(q,j3))))/3.0;
    return exp(-mean/0.5);

}

__device__ float d_opposite_side(float2 *p,int i){
    float2 a;
    a.x = p[(i+1)%3].x - p[(i+2)%3].x;
    a.y = p[(i+1)%3].y - p[(i+2)%3].y;
    return sqrt(pow(a.x,2)+pow(a.y,2));
}

__device__ float d_sim_ratios(float2 *p, float2 *q, int2 *idx){
    int i1 = idx[0].x;
    int j1 = idx[0].y;
    int i2 = idx[1].x;
    int j2 = idx[1].y;
    int i3 = idx[2].x;
    int j3 = idx[2].y;
    float R1, R2, R3;
    R1 = d_opposite_side(p,i1)/d_opposite_side(q,j1);
    R2 = d_opposite_side(p,i2)/d_opposite_side(q,j2);
    R3 = d_opposite_side(p,i3)/d_opposite_side(q,j3);
    float mean = (R1 + R2 + R3)/3.0;
    return ((R1-mean)+(R2-mean)+(R3-mean))/3.0;
}



__device__ float d_similarity(float2 *p, float2 *q){

    int2 perms_0[3],perms_1[3],perms_2[3];
    int2 perms_3[3],perms_4[3],perms_5[3];
    perms_0[0].x = 0;
    perms_0[0].y = 0;
    perms_0[1].x = 1;
    perms_0[1].y = 1;
    perms_0[2].x = 2;
    perms_0[2].y = 2;

    perms_1[0].x = 0;
    perms_1[0].y = 0;
    perms_1[1].x = 1;
    perms_1[1].y = 1;
    perms_1[2].x = 2;
    perms_1[2].y = 2;

    perms_2[0].x = 0;
    perms_2[0].y = 0;
    perms_2[1].x = 1;
    perms_2[1].y = 1;
    perms_2[2].x = 2;
    perms_2[2].y = 2;

    perms_3[0].x = 0;
    perms_3[0].y = 0;
    perms_3[1].x = 1;
    perms_3[1].y = 1;
    perms_3[2].x = 2;
    perms_3[2].y = 2;

    perms_4[0].x = 0;
    perms_4[0].y = 0;
    perms_4[1].x = 1;
    perms_4[1].y = 1;
    perms_4[2].x = 2;
    perms_4[2].y = 2;

    perms_5[0].x = 0;
    perms_5[0].y = 0;
    perms_5[1].x = 1;
    perms_5[1].y = 1;
    perms_5[2].x = 2;
    perms_5[2].y = 2;

    float sim_a = d_sim_angles(p,q,perms_0);
    float sim_r = d_sim_ratios(p,q,perms_0);
    return sim_a;
}


__global__ void d_hyperedges (int *edges1, int *edges2,
        float *kp1, float *kp2,
        float *desc1, float *desc2, double c1,
        double c2, double c3, double thresholding,
        int edges1Size, int edges2Size, float *matches){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float2 *p,*q;
    p = (float2*) malloc(3*sizeof(float2));
    q = (float2*) malloc(3*sizeof(float2));

    if (i < edges1Size){
        for (int j = 0; j < edges2Size; j++) {
            p[0].x = kp1[(edges1[i*3+0])*2+0];
            p[0].y = kp1[(edges1[i*3+0])*2+1];
            p[1].x = kp1[(edges1[i*3+1])*2+0];
            p[1].y = kp1[(edges1[i*3+1])*2+1];
            p[2].x = kp1[(edges1[i*3+2])*2+0];
            p[2].y = kp1[(edges1[i*3+2])*2+1];
            q[0].x = kp2[(edges2[j*3+0])*2+0];
            q[0].y = kp2[(edges2[j*3+0])*2+1];
            q[1].x = kp2[(edges2[j*3+1])*2+0];
            q[1].y = kp2[(edges2[j*3+1])*2+1];
            q[2].x = kp2[(edges2[j*3+2])*2+0];
            q[2].y = kp2[(edges2[j*3+2])*2+1];
            float sim_a = d_similarity(p,q);
            matches[i*edges2Size+j] = p[0].x;
        }
    }

    free(p);free(q);

}
