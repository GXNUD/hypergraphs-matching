#include <stdio.h>
#include <stdlib.h>

#define SIGMA 0.5
#define SIZE_PERMS 3

typedef struct float2
{
    float x,y;
}f2;

typedef struct int2
{
    int x,y;
}i2;



typedef struct MatchSimilarity
{
    float sim_a, sim_r, sim_d, sim;
    int2 permutation[3];
}MatchS;

typedef struct beforeMatches
{
    int bestIndex_j;
    int2 edge_match_indices[3];
    float s_ang, s_rat, s_desc, max_similarity;

}bMatchS;

float d_angle(float2 *p, int i){
    float2 a;
    float2 b;
    a.x = p[i].x - p[(i+1)%3].x;
    a.y = p[i].y - p[(i+1)%3].y;
    b.x = p[i].x - p[(i+2)%3].x;
    b.y = p[i].y - p[(i+2)%3].y;
    float dot = a.x*b.x+a.y*b.y;
    float angle = acosf(dot/sqrtf((a.x*a.x)+(a.y*a.y))
            /sqrtf((b.x*b.x)+(b.y*b.y)));
    return angle;
}

float d_sim_angles(float2 *p, float2 *q, int2 *idx){
    int i1 = idx[0].x;
    int j1 = idx[0].y;
    int i2 = idx[1].x;
    int j2 = idx[1].y;
    int i3 = idx[2].x;
    int j3 = idx[2].y;

    float mean = (fabs(sinf(d_angle(p,i1))-sinf(d_angle(q,j1))) +
        fabs(sinf(d_angle(p,i2))-sinf(d_angle(q,j2))) +
        fabs(sinf(d_angle(p,i3))-sinf(d_angle(q,j3))))/3.0;

    return expf(-mean/0.5);

}


float d_opposite_side(float2 *p,int i){
    float2 a;
    a.x = p[(i+1)%3].x - p[(i+2)%3].x;
    a.y = p[(i+1)%3].y - p[(i+2)%3].y;
    return sqrtf(powf(a.x,2)+powf(a.y,2));
}

float d_sim_ratios(float2 *p, float2 *q, int2 *idx){
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
    float standard = sqrtf((powf((R1-mean),2)+(powf(R2-mean,2))
            +(powf(R3-mean,2)))/3.0);
    return expf(-standard/0.5);
}

float d_sim_desc(float *dp, float *dq, int2 *idx){
    int i1 = idx[0].x;
    int j1 = idx[0].y;
    int i2 = idx[1].x;
    int j2 = idx[1].y;
    int i3 = idx[2].x;
    int j3 = idx[2].y;
    float a = 0.0;
    float b = 0.0;
    float c = 0.0;
    float mean;

    for (int i = 0; i < 64; i++) {
        a = a + powf(dp[i1*64+i] - dq[j1*64+i],2);
        b = b + powf(dp[i2*64+i] - dq[j2*64+i],2);
        c = c + powf(dp[i3*64+i] - dq[j3*64+i],2);
    }
    a = sqrtf(a); b = sqrtf(b); c = sqrtf(c);
    mean = (a+b+c)/3.0;
    return expf(-mean/0.5);

}

MatchSimilarity d_similarity(float2 *p, float2 *q, float *dp, float *dq,
        float cang, float crat, float cdesc){

    int2 perms_0[3],perms_1[3],perms_2[3];
    int2 perms_3[3],perms_4[3],perms_5[3];
    int2 point_match[3];
    perms_0[0].x = 0;
    perms_0[0].y = 0;
    perms_0[1].x = 1;
    perms_0[1].y = 1;
    perms_0[2].x = 2;
    perms_0[2].y = 2;

    perms_1[0].x = 0;
    perms_1[0].y = 0;
    perms_1[1].x = 1;
    perms_1[1].y = 2;
    perms_1[2].x = 2;
    perms_1[2].y = 1;

    perms_2[0].x = 0;
    perms_2[0].y = 1;
    perms_2[1].x = 1;
    perms_2[1].y = 0;
    perms_2[2].x = 2;
    perms_2[2].y = 2;

    perms_3[0].x = 0;
    perms_3[0].y = 1;
    perms_3[1].x = 1;
    perms_3[1].y = 2;
    perms_3[2].x = 2;
    perms_3[2].y = 0;

    perms_4[0].x = 0;
    perms_4[0].y = 2;
    perms_4[1].x = 1;
    perms_4[1].y = 0;
    perms_4[2].x = 2;
    perms_4[2].y = 1;

    perms_5[0].x = 0;
    perms_5[0].y = 2;
    perms_5[1].x = 1;
    perms_5[1].y = 1;
    perms_5[2].x = 2;
    perms_5[2].y = 0;

    float s = cang + crat + cdesc;
    cang = cang/s;
    crat = crat/s;
    cdesc = cdesc/s;
    float sim = -1000.0;
    float d_sim_a, d_sim_r, d_sim_d, d_sim, sim_a, sim_r, sim_d;

    d_sim_a = d_sim_angles(p,q,perms_0);
    d_sim_r = d_sim_ratios(p,q,perms_0);
    d_sim_d = d_sim_desc(dp,dq,perms_0);
    d_sim = cang * d_sim_a + crat * d_sim_r + cdesc * d_sim_d;
    if(d_sim>sim){
        #pragma unroll
        for (int ll = 0; ll<SIZE_PERMS; ll++) {
            point_match[ll].x = perms_0[ll].x;
            point_match[ll].y = perms_0[ll].y;
        }
        sim_a = d_sim_a;
        sim_r = d_sim_r;
        sim_d = d_sim_d;
        sim = d_sim;
    }

    d_sim_a = d_sim_angles(p,q,perms_1);
    d_sim_r = d_sim_ratios(p,q,perms_1);
    d_sim_d = d_sim_desc(dp,dq,perms_1);
    d_sim = cang * d_sim_a + crat * d_sim_r + cdesc * d_sim_d;
    if(d_sim>sim){
        #pragma unroll
        for (int ll = 0; ll<SIZE_PERMS; ll++) {
            point_match[ll].x = perms_1[ll].x;
            point_match[ll].y = perms_1[ll].y;
        }
        sim_a = d_sim_a;
        sim_r = d_sim_r;
        sim_d = d_sim_d;
        sim = d_sim;
    }

    d_sim_a = d_sim_angles(p,q,perms_2);
    d_sim_r = d_sim_ratios(p,q,perms_2);
    d_sim_d = d_sim_desc(dp,dq,perms_2);
    d_sim = cang * d_sim_a + crat * d_sim_r + cdesc * d_sim_d;
    if(d_sim>sim){
         #pragma unroll
        for (int ll = 0; ll<SIZE_PERMS; ll++) {
            point_match[ll].x = perms_2[ll].x;
            point_match[ll].y = perms_2[ll].y;
        }
        sim_a = d_sim_a;
        sim_r = d_sim_r;
        sim_d = d_sim_d;
        sim = d_sim;
   }

    d_sim_a = d_sim_angles(p,q,perms_3);
    d_sim_r = d_sim_ratios(p,q,perms_3);
    d_sim_d = d_sim_desc(dp,dq,perms_3);
    d_sim = cang * d_sim_a + crat * d_sim_r + cdesc * d_sim_d;
    if(d_sim>sim){
        #pragma unroll
        for (int ll = 0; ll<SIZE_PERMS; ll++) {
            point_match[ll].x = perms_3[ll].x;
            point_match[ll].y = perms_3[ll].y;
        }
        sim_a = d_sim_a;
        sim_r = d_sim_r;
        sim_d = d_sim_d;
        sim = d_sim;
    }

    d_sim_a = d_sim_angles(p,q,perms_4);
    d_sim_r = d_sim_ratios(p,q,perms_4);
    d_sim_d = d_sim_desc(dp,dq,perms_4);
    d_sim = cang * d_sim_a + crat * d_sim_r + cdesc * d_sim_d;
    if(d_sim>sim){
         #pragma unroll
        for (int ll = 0; ll<SIZE_PERMS; ll++) {
            point_match[ll].x = perms_4[ll].x;
            point_match[ll].y = perms_4[ll].y;
        }
        sim_a = d_sim_a;
        sim_r = d_sim_r;
        sim_d = d_sim_d;
        sim = d_sim;
   }

    d_sim_a = d_sim_angles(p,q,perms_5);
    d_sim_r = d_sim_ratios(p,q,perms_5);
    d_sim_d = d_sim_desc(dp,dq,perms_5);
    d_sim = cang * d_sim_a + crat * d_sim_r + cdesc * d_sim_d;
    if(d_sim>sim){
        #pragma unroll
        for (int ll = 0; ll<SIZE_PERMS; ll++) {
            point_match[ll].x = perms_5[ll].x;
            point_match[ll].y = perms_5[ll].y;
        }
        sim_a = d_sim_a;
        sim_r = d_sim_r;
        sim_d = d_sim_d;
        sim = d_sim;
  }

    MatchSimilarity finalSimilarity;
    finalSimilarity.sim_a = sim_a;
    finalSimilarity.sim_r = sim_r;
    finalSimilarity.sim_d = sim_d;
    finalSimilarity.sim = sim;
    #pragma unroll
    for (int ll = 0; ll<SIZE_PERMS; ll++) {
        finalSimilarity.permutation[ll].x = point_match[ll].x;
        finalSimilarity.permutation[ll].y = point_match[ll].y;
    }
    return finalSimilarity;
}


void d_hyperedges (int *edges1, int *edges2,
        float *kp1, float *kp2,
        float *desc1, float *desc2, int desc1Rows,
        int desc1Cols, int desc2Rows, int desc2Cols, float cang,
        float crat, float cdesc, float thresholding,
        int edges1Size, int edges2Size, beforeMatches *before_matches, float *tests){

    float2 *p,*q;
    float *desc_p, *desc_q;
    p = (float2*) malloc(3*sizeof(float2));
    q = (float2*) malloc(3*sizeof(float2));

    desc_p = (float*)malloc(desc1Cols*sizeof(float)*3);
    desc_q = (float*)malloc(desc2Cols*sizeof(float)*3);

    float max_similarity, s_ang, s_ratios, s_desc;
    int best_index=-1;
    int2 edge_match_indices[3];
    int j =0;

    MatchSimilarity finalSimilarity;
    float t_ang,t_rat,t_desc,t_sim;
    for(int i =0;i < edges1Size;i++){
        max_similarity = -1000.0;;
        for (j = 0; j < edges2Size; j++) {
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

            tests[i*edges2Size+j] = desc_q[2*desc2Cols+0];

            finalSimilarity = d_similarity(p,q,desc_p,desc_q,
                    cang,crat,cdesc);

            if(finalSimilarity.sim > max_similarity){
                best_index = j;
                max_similarity = finalSimilarity.sim;
                s_ang = finalSimilarity.sim_a;
                s_ratios = finalSimilarity.sim_r;
                s_desc = finalSimilarity.sim_d;
                for (int ii = 0; ii < 3 ; ii++) {
                    int p_i = finalSimilarity.permutation[ii].x;
                    int q_i = finalSimilarity.permutation[ii].y;
                    edge_match_indices[ii].x = edges1[i*3+p_i];
                    edge_match_indices[ii].y = edges2[j*3+q_i];
                }

                before_matches[i].bestIndex_j = best_index;
                before_matches[i].max_similarity = max_similarity;
                before_matches[i].s_ang = s_ang;
                before_matches[i].s_rat = s_ratios;
                before_matches[i].s_desc = s_desc;
                before_matches[i].edge_match_indices[0].x = edge_match_indices[0].x;
                before_matches[i].edge_match_indices[0].y = edge_match_indices[0].y;
                before_matches[i].edge_match_indices[1].x = edge_match_indices[1].x;
                before_matches[i].edge_match_indices[1].y = edge_match_indices[1].y;
                before_matches[i].edge_match_indices[2].x = edge_match_indices[2].x;
                before_matches[i].edge_match_indices[2].y = edge_match_indices[2].y;
            }

        }


    }

    free(p);free(q);free(desc_p);free(desc_q);

}
