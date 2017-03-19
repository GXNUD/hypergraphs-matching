#include <stdio.h>
#include <stdlib.h>

#define SIGMA 0.5

typedef struct MatchSimilarity
{
    float sim_a, sim_r, sim_d, sim;
    int2 *permutation;
}MatchS;

typedef struct beforeMatches
{
    int bestIndex_j;
    int2 edge_match_indices[3];

}bMatchS;

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
    float standard = (pow((R1-mean),2)+(pow(R2-mean,2))+(pow(R3-mean,2)))/3.0;
    return exp(-standard/0.5);
}

__device__ float d_sim_desc(float *dp, float *dq, int2 *idx){
    int i1 = idx[0].x;
    int j1 = idx[0].y;
    int i2 = idx[1].x;
    int j2 = idx[1].y;
    int i3 = idx[2].x;
    int j3 = idx[2].y;
    float a = dp[i1]-dq[j1];
    float b = dp[i2]-dq[j2];
    float c = dp[i3]-dq[j3];
    a = sqrt(pow(a,2));
    b = sqrt(pow(b,2));
    c = sqrt(pow(b,2));

    return ((a+b+c)/3.0)/0.5;

}

__device__ MatchSimilarity d_similarity(float2 *p, float2 *q, float *dp, float *dq,
        float cang, float crat, float cdesc){

    int2 perms_0[3],perms_1[3],perms_2[3];
    int2 perms_3[3],perms_4[3],perms_5[3];
    int2 *point_match;
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

    float s = cang + crat + cdesc;
    cang /= s;
    crat /= s;
    cdesc /= s;
    float sim = -1E30;
    float _sim_a, _sim_r, _sim_d, _sim, sim_a, sim_r, sim_d;

    _sim_a = d_sim_angles(p,q,perms_0);
    _sim_r = d_sim_ratios(p,q,perms_0);
    _sim_d = d_sim_desc(dp,dq,perms_0);
    _sim = cang * _sim_a + crat * _sim_r + cdesc * _sim_d;
    if(_sim>sim){
        point_match = perms_0;
        sim_a = _sim_a;
        sim_r = _sim_r;
        sim_d = _sim_d;
        sim = _sim;
    }

    _sim_a = d_sim_angles(p,q,perms_1);
    _sim_r = d_sim_ratios(p,q,perms_1);
    _sim_d = d_sim_desc(dp,dq,perms_1);
    _sim = cang * sim_a + crat * sim_r + cdesc * sim_d;
    if(_sim>sim){
        point_match = perms_1;
        sim_a = _sim_a;
        sim_r = _sim_r;
        sim_d = _sim_d;
        sim = _sim;
    }

    _sim_a = d_sim_angles(p,q,perms_2);
    _sim_r = d_sim_ratios(p,q,perms_2);
    _sim_d = d_sim_desc(dp,dq,perms_2);
    _sim = cang * sim_a + crat * sim_r + cdesc * sim_d;
    if(_sim>sim){
        point_match = perms_2;
        sim_a = _sim_a;
        sim_r = _sim_r;
        sim_d = _sim_d;
        sim = _sim;
   }

    _sim_a = d_sim_angles(p,q,perms_3);
    _sim_r = d_sim_ratios(p,q,perms_3);
    _sim_d = d_sim_desc(dp,dq,perms_3);
    _sim = cang * sim_a + crat * sim_r + cdesc * sim_d;
    if(_sim>sim){
        point_match = perms_3;
        sim_a = _sim_a;
        sim_r = _sim_r;
        sim_d = _sim_d;
        sim = _sim;
    }

    _sim_a = d_sim_angles(p,q,perms_4);
    _sim_r = d_sim_ratios(p,q,perms_4);
    _sim_d = d_sim_desc(dp,dq,perms_4);
    _sim = cang * sim_a + crat * sim_r + cdesc * sim_d;
    if(_sim>sim){
        point_match = perms_4;
        sim_a = _sim_a;
        sim_r = _sim_r;
        sim_d = _sim_d;
        sim = _sim;
   }

    _sim_a = d_sim_angles(p,q,perms_5);
    _sim_r = d_sim_ratios(p,q,perms_5);
    _sim_d = d_sim_desc(dp,dq,perms_5);
    _sim = cang * sim_a + crat * sim_r + cdesc * sim_d;
    if(_sim>sim){
        point_match = perms_5;
        sim_a = _sim_a;
        sim_r = _sim_r;
        sim_d = _sim_d;
        sim = _sim;
  }

    MatchSimilarity finalSimilarity;

    finalSimilarity.sim_a = sim_a;
    finalSimilarity.sim_r = sim_r;
    finalSimilarity.sim_d = sim_d;
    finalSimilarity.sim = sim;
    finalSimilarity.permutation = point_match;

    return finalSimilarity;
}


__global__ void d_hyperedges (int *edges1, int *edges2,
        float *kp1, float *kp2,
        float *desc1, float *desc2, int desc1Rows,
        int desc1Cols, int desc2Rows, int desc2Cols, float cang,
        float crat, double cdesc, double thresholding,
        int edges1Size, int edges2Size, beforeMatches *before_matches){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float2 *p,*q;
    float *desc_p, *desc_q;
    p = (float2*) malloc(3*sizeof(float2));
    q = (float2*) malloc(3*sizeof(float2));

    desc_p = (float*)malloc(desc1Cols*sizeof(float)*3);
    desc_q = (float*)malloc(desc2Cols*sizeof(float)*3);

    float best_index, max_similarity, s_ang, s_ratios, s_desc;
    int2 edge_match_indices[3];

    if (i < edges1Size){
        max_similarity = -1E30;
        for (int j = 0; j < edges2Size; j++) {
            //keyPoints
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
            //////////////////////////////////
            //Descriptors
            for (int ii = 0; ii < desc1Cols; ii++) {
                desc_p[0*desc1Cols+ii] = desc1[(edges1[i*3+0])*desc1Cols+ii];
                desc_p[1*desc1Cols+ii] = desc1[(edges1[i*3+1])*desc1Cols+ii];
                desc_p[2*desc1Cols+ii] = desc1[(edges1[i*3+2])*desc1Cols+ii];
            }
            for (int ii = 0; ii < desc2Cols; ii++) {
                desc_q[0*desc2Cols+ii] = desc2[(edges2[j*3+0])*desc2Cols+ii];
                desc_q[1*desc2Cols+ii] = desc2[(edges2[j*3+1])*desc2Cols+ii];
                desc_q[2*desc2Cols+ii] = desc2[(edges2[j*3+2])*desc2Cols+ii];
            }

            MatchSimilarity finalSimilarity = d_similarity(p,q,desc_p,desc_q,cang,crat,cdesc);
            if(finalSimilarity.sim > max_similarity){
                best_index = j;
                max_similarity = finalSimilarity.sim;
                s_ang = finalSimilarity.sim_a;
                s_ratios = finalSimilarity.sim_r;
                s_desc = finalSimilarity.sim_d;
                for (int ii = 0; ii < 3 ; ii++) {
                    int p_i = finalSimilarity.permutation[ii].x;
                    int q_i = finalSimilarity.permutation[ii].y;
                    edge_match_indices[ii].x = edges1[p_i*3+ii];
                    edge_match_indices[ii].y = edges2[q_i*3+ii];
                }
            }
            //matches[i*edges2Size+j] = (float)edge_match_indices[0].y;
        }
        before_matches[i].bestIndex_j = best_index;
        before_matches[i].edge_match_indices[0].x = edge_match_indices[0].x;
        before_matches[i].edge_match_indices[0].y = edge_match_indices[0].y;

        before_matches[i].edge_match_indices[1].x = edge_match_indices[1].x;
        before_matches[i].edge_match_indices[1].y = edge_match_indices[1].y;

        before_matches[i].edge_match_indices[2].x = edge_match_indices[2].x;
        before_matches[i].edge_match_indices[2].y = edge_match_indices[2].y;

    }

    free(p);free(q);free(desc_p);free(desc_q);

}
