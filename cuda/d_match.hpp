#include <stdio.h>
#include <stdlib.h>

#define SIGMA 0.5

__device__ int* d_getCombination(int n, int r){
    bool *v = (bool*)malloc(sizeof(bool)*n);
    int *combinations = (int*)malloc(sizeof(int)*3*2);
    v[0] = true; v[1] = true; v[2] = false;
    for (int i = 0; i < n; i++) {
        if(v[i]){
            combinations[0*2+i] = i;
        }
    }
    v[0] = true; v[1] = false; v[2] = true;
    for (int i = 0; i < n; i++) {
        if(v[i]){
            combinations[1*2+i] = i;
        }
    }
    v[0] = false; v[1] = true; v[2] = true;
    for (int i = 0; i < n; i++) {
        if(v[i]){
            combinations[2*2+i] = i;
        }
    }

    free(v);
    return combinations;

}

__device__ float d_ratios(float *p, float *q){
    float *idx_perm = d_getCombination(3,2);
    float *sides_p = (float*)malloc(sizeof(float)*3);
    float *sides_q = (float*)malloc(sizeof(float)*3);
    for (int k = 0; k < 3; k++) {
        int i = idx_perm[k*2+0];
        int j = idx_perm[k*2+1];
        int p_x = p[i*2+0] - p[j*2+0];
        int p_y = p[i*2+1] - p[j*2+1];
        int q_x = q[i*2+0] - q[j*2+0];
        int q_y = q[i*2+1] - q[j*2+1];
        float side_p = sqrt(pow(p_x,2)+pow(p_y,2));
        float side_q = sqrt(pow(q_x,2)+pow(q_y,2));
        sides_p[i] = side_p;
        sides_q[i] = side_q;
    }

    free(sides_p);free(sides_q);free(idx_perm);

}

__device__ float vectorsAngleSin(float pivot_x, float pivot_y, float p_x, float p_y,
        float q_x, float q_y){
    float v1_x = p_x - pivot_x;
    float v1_y = p_y - pivot_y;
    float v2_x = q_x - pivot_x;
    float v2_y = q_y - pivot_y;
    float dot = v1_x*v2_x+v1_y*v2_y;
    float angle = acos(dot/sqrt((v1_x*v1_x)+(v1_y*v1_y))/sqrt((v2_x*v2_x)+(v2_y*v2_y)));
    return sin(angle);
}

__device__ float* d_getAnglesSin(float *p){
    float p1_x = p[0*2+0];
    float p1_y = p[0*2+1];
    float p2_x = p[1*2+0];
    float p2_y = p[1*2+1];
    float p3_x = p[2*2+0];
    float p3_y = p[2*2+1];
    float *sines = (float*)malloc(3*sizeof(float));
    sines[0] = vectorsAngleSin(p1_x,p1_y,p2_x,p2_y,p3_x,p3_y);
    sines[1] = vectorsAngleSin(p2_x,p2_y,p1_x,p1_y,p3_x,p3_y);
    sines[2] = vectorsAngleSin(p3_x,p3_y,p1_x,p1_y,p2_x,p2_y);
    return sines;
}

__device__ float d_sim_angles(float *p, float *q){
    float *sines1;// = (float*)malloc(3*sizeof(float));
    float *sines2;// = (float*)malloc(3*sizeof(float));
    sines1 = d_getAnglesSin(p);
    sines2 = d_getAnglesSin(q);
    float min_diff_between_sin = 1E30;

    float sum = 0.0;
    sum = (fabs(sines1[0] - sines2[0])) + (fabs(sines1[1] - sines2[1])) +
        (fabs(sines1[2]-sines2[2]));
    min_diff_between_sin = min(min_diff_between_sin,sum);

    sum = (fabs(sines1[0] - sines2[0])) + (fabs(sines1[2] - sines2[1])) +
        (fabs(sines1[1]-sines2[2]));
    min_diff_between_sin = min(min_diff_between_sin,sum);

    sum = (fabs(sines1[1] - sines2[0])) + (fabs(sines1[0] - sines2[1])) +
        (fabs(sines1[2]-sines2[2]));
    min_diff_between_sin = min(min_diff_between_sin,sum);

    sum = (fabs(sines1[1] - sines2[0])) + (fabs(sines1[2] - sines2[1])) +
        (fabs(sines1[0]-sines2[2]));
    min_diff_between_sin = min(min_diff_between_sin,sum);

    sum = (fabs(sines1[2] - sines2[0])) + (fabs(sines1[0] - sines2[1])) +
        (fabs(sines1[1]-sines2[2]));
    min_diff_between_sin = min(min_diff_between_sin,sum);

    sum = (fabs(sines1[2] - sines2[0])) + (fabs(sines1[1] - sines2[1])) +
        (fabs(sines1[0]-sines2[2]));
    min_diff_between_sin = min(min_diff_between_sin,sum);

    free(sines1); free(sines2);

    return exp(-min_diff_between_sin / SIGMA);

}


__global__ void d_hyperedges (int *edges1, int *edges2,
        float *kp1, float *kp2,
        float *desc1, float *desc2, double c1,
        double c2, double c3, double thresholding,
        int edges1Size, int edges2Size, float *matches){

    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    float *e1_points,*e2_points;
    e1_points = (float*) malloc(6*sizeof(float));
    e2_points = (float*) malloc(6*sizeof(float));

    if ((i < edges1Size) && (j < edges2Size)){
     //   float p[SIZE_POINTS], q[SIZE_POINTS];
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

        float sim_angles = d_sim_angles(e1_points, e2_points);
        matches[i*edges2Size+j] = sim_angles;

    }

    free(e1_points);free(e2_points);
}
