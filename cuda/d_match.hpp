#include <stdio.h>
#include <stdlib.h>

#define SIZE 3
#define SIZE_POINTS 6
/*#define D_DOT_1 (p[2]-p[0])*(p[4]-p[0])+(p[2]-p[1])*(p[5]-p[1])
#define NORMV1_1 sqrt((p[2]-p[0])*(p[2]-p[0])+(p[2]-p[1])*(p[2]-p[1]))
#define NORMV2_1 sqrt((p[4]-p[0])*(p[4]-p[0])+(p[5]-p[1])*(p[5]-p[1]))
#define D_DOT_2 (p[0]-p[2])*(p[4]-p[2])+(p[1]-p[3])*(p[5]-p[3])
#define NORMV1_2 sqrt((p[0]-p[2])*(p[0]-p[2])+(p[1]-p[3])*(p[1]-p[3]))
#define NORMV2_2 sqrt((p[4]-p[2])*(p[4]-p[2])+(p[5]-p[3])*(p[5]-p[3]))
#define D_DOT_3 (p[0]-p[4])*(p[2]-p[4])+(p[1]-p[5])*(p[3]-p[5])
#define NORMV1_3 sqrt((p[0]-p[4])*(p[0]-p[4])+(p[1]-p[5])*(p[1]-p[5]))
#define NORMV2_3 sqrt((p[2]-p[4])*(p[2]-p[4])+(p[3]-p[4])*(p[3]-p[4]))
#define D_DOT_4 (q[2]-q[0])*(q[4]-q[0])+(q[2]-q[1])*(q[5]-q[1])
#define NORMV1_4 sqrt((q[2]-q[0])*(q[2]-q[0])+(q[2]-q[1])*(q[2]-q[1]))
#define NORMV2_4 sqrt((q[4]-q[0])*(q[4]-q[0])+(q[5]-q[1])*(q[5]-p[1]))
#define D_DOT_5 (q[0]-q[2])*(q[4]-q[2])+(q[1]-q[3])*(q[5]-q[3])
#define NORMV1_5 sqrt((q[0]-q[2])*(q[0]-q[2])+(q[1]-q[3])*(q[1]-q[3]))
#define NORMV2_5 sqrt((q[4]-q[2])*(q[4]-q[2])+(q[5]-q[3])*(q[5]-q[3]))
#define D_DOT_6 (q[0]-q[4])*(q[2]-q[4])+(q[1]-q[5])*(q[3]-q[5])
#define NORMV1_6 sqrt((q[0]-q[4])*(q[0]-q[4])+(q[1]-q[5])*(q[1]-q[5]))
#define NORMV2_6 sqrt((q[2]-q[4])*(q[2]-q[4])+(q[3]-q[4])*(q[3]-q[4]))
#define sines1_0 sin(acos(D_DOT_1/NORMV1_1/NORMV2_1))
#define sines1_1 sin(acos(D_DOT_2/NORMV1_2/NORMV2_2))
#define sines1_2 sin(acos(D_DOT_3/NORMV1_3/NORMV2_3))
#define sines2_0 sin(acos(D_DOT_4/NORMV1_4/NORMV2_4))
#define sines2_1 sin(acos(D_DOT_5/NORMV1_5/NORMV2_5))
#define sines2_2 sin(acos(D_DOT_6/NORMV1_6/NORMV2_6))*/

#define SIGMA 0.5


__device__ float vectorsAngleSin(float pivot_x, float pivot_y, float p_x, float p_y,
        float q_x, float q_y){
    float v1_x = p_x - pivot_x;
    float v1_y = p_y - pivot_y;
    float v2_x = q_x - pivot_x;
    float v2_y = q_y - pivot_y;
    float dot = v1_x*v2_x+v1_y+v2_y;
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
    float *sines1 = (float*)malloc(3*sizeof(float));
    float *sines2 = (float*)malloc(3*sizeof(float));
    sines1 = d_getAnglesSin(p);
    sines2 = d_getAnglesSin(q);
    float min_diff_between_sin = 1E30;

    float sum = 0.0;
    sum = sum + (fabs(sines1[0] - sines2[0])) + (fabs(sines1[1] - sines2[1])) +
        (fabs(sines1[2]-sines2[2]));
    min_diff_between_sin = min(min_diff_between_sin,sum);

    sum = sum + (fabs(sines1[0] - sines2[0])) + (fabs(sines1[2] - sines2[1])) +
        (fabs(sines1[1]-sines2[2]));
    min_diff_between_sin = min(min_diff_between_sin,sum);

    sum = sum + (fabs(sines1[1] - sines2[0])) + (fabs(sines1[0] - sines2[1])) +
        (fabs(sines1[2]-sines2[2]));
    min_diff_between_sin = min(min_diff_between_sin,sum);

    sum = sum + (fabs(sines1[1] - sines2[0])) + (fabs(sines1[2] - sines2[1])) +
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
    e1_points = (float*) malloc(SIZE_POINTS*sizeof(float));
    e2_points = (float*) malloc(SIZE_POINTS*sizeof(float));

    if ((i < edges1Size) && (j < edges2Size)){
     //   float p[SIZE_POINTS], q[SIZE_POINTS];
        e1_points[0*2+0] = kp1[edges1[i*SIZE+0]*2+0];
        e1_points[0*2+1] = kp1[edges1[i*SIZE+0]*2+1];
        e1_points[1*2+0] = kp1[edges1[i*SIZE+1]*2+0];
        e1_points[1*2+1] = kp1[edges1[i*SIZE+1]*2+1];
        e1_points[2*2+0] = kp1[edges1[i*SIZE+2]*2+0];
        e1_points[2*2+1] = kp1[edges1[i*SIZE+2]*2+1];
        e2_points[0*2+0] = kp2[edges2[j*SIZE+0]*2+0];
        e2_points[0*2+1] = kp2[edges2[j*SIZE+0]*2+1];
        e2_points[1*2+0] = kp2[edges2[j*SIZE+1]*2+0];
        e2_points[1*2+1] = kp2[edges2[j*SIZE+1]*2+1];
        e2_points[2*2+0] = kp2[edges2[j*SIZE+2]*2+0];
        e2_points[2*2+1] = kp2[edges2[j*SIZE+2]*2+1];

        float sim_angles = d_sim_angles(e1_points, e2_points);

       /* float sines1[3];
        float sines2[3];
        sines1[0] = sin(acos(D_DOT_1/NORMV1_1/NORMV2_1));
        sines1[1] = sin(acos(D_DOT_2/NORMV1_2/NORMV2_2));
        sines1[2] = sin(acos(D_DOT_3/NORMV1_3/NORMV2_3));
        sines2[0] = sin(acos(D_DOT_4/NORMV1_4/NORMV2_4));
        sines2[1] = sin(acos(D_DOT_5/NORMV1_5/NORMV2_5));
        sines2[2] = sin(acos(D_DOT_6/NORMV1_6/NORMV2_6));*/
        /*double sum1=fabs(sines1_0-sines2_0)+fabs(sines1_1-sines2_1)+fabs(sines1_2-sines2_2);
        double sum2=fabs(sines1_0-sines2_0)+fabs(sines1_2-sines2_2)+fabs(sines1_1-sines2_1);
        double sum3=fabs(sines1_1-sines2_1)+fabs(sines1_0-sines2_0)+fabs(sines1_2-sines2_2);
        double sum4=fabs(sines1_1-sines2_1)+fabs(sines1_2-sines2_2)+fabs(sines1_1-sines2_1);
        double sum5=fabs(sines1_2-sines2_2)+fabs(sines1_0-sines2_0)+fabs(sines1_1-sines2_1);
        double sum6=fabs(sines1_2-sines2_2)+fabs(sines1_1-sines2_1)+fabs(sines1_2-sines2_2);*/

        /*double sum1=fabs(sines1[0]-sines2[0])+fabs(sines1[1]-sines2[1])+fabs(sines1[2]-sines2[2]);
        double sum2=fabs(sines1[0]-sines2[0])+fabs(sines1[2]-sines2[2])+fabs(sines1[1]-sines2[1]);
        double sum3=fabs(sines1[1]-sines2[1])+fabs(sines1[0]-sines2[0])+fabs(sines1[2]-sines2[2]);
        double sum4=fabs(sines1[1]-sines2[1])+fabs(sines1[2]-sines2[2])+fabs(sines1[1]-sines2[1]);
        double sum5=fabs(sines1[2]-sines2[2])+fabs(sines1[0]-sines2[0])+fabs(sines1[1]-sines2[1]);
        double sum6=fabs(sines1[2]-sines2[2])+fabs(sines1[1]-sines2[1])+fabs(sines1[2]-sines2[2]);*/

        /*double minimo;
        if(sum1<=sum2)
            minimo=sum1;
        else
            minimo=sum2;
        if(sum3<minimo)
            minimo=sum3;
        if(sum4<minimo)
            minimo=sum4;
        if(sum5<minimo)
            minimo=sum5;
        if(sum6<minimo)
            minimo=sum6;*/

        //sines = exp(-min/SIGMA);
        if(i==0&&j==0)
            matches[i*edges2Size+j] = sim_angles;
    }
    free(e1_points);free(e2_points);
}

/*__global__ void calculate_e1_points(int *edges1, float *kp1, int edges1Size, float *e1_points){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j;
    if(i<edges1Size){
        e1_points[i*6+0] = edges1[0*SIZE+0];
        e1_points[i*6+1] = kp1[(edges1[i*SIZE+0])*2+1];
        e1_points[i*6+2] = kp1[(edges1[i*SIZE+1])*2+0];
        e1_points[i*6+3] = kp1[(edges1[i*SIZE+1])*2+1];
        e1_points[i*6+4] = kp1[(edges1[i*SIZE+2])*2+0];
        e1_points[i*6+5] = kp1[(edges1[i*SIZE+2])*2+1];
    }
}*/

