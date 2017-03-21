#include <vector>
#include <cmath>
#include <algorithm>
#include "opencv2/core/core.hpp"

#define SIGMA 0.5

using namespace cv;
using namespace std;

typedef pair<int,int> perm_t;
struct MatchSimilarity {
    vector<perm_t> match_indices; // pairs of indices of best configuration between the hyperedges
    double global_sim;
    double angles_sim;
    double ratios_sim;
    double desc_sim;
};

vector<vector<pair<int, int>>> PERMS = {
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)}
};

float mean(float x1, float x2, float x3) {
  float m = (x1 + x2 + x3) / 3;
  return m;
}

// Standar deviation
float stdDev(float x1, float x2, float x3) {
  float m = mean(x1, x2, x3);
  float t1 = (x1 - m) * (x1 - m);
  float t2 = (x2 - m) * (x2 - m);
  float t3 = (x3 - m) * (x3 - m);
  float s = sqrt((t1 + t2 + t3) / 3);
  return s;
}

float getAngle(vector<Point2f> &triangle, int pivot_idx) {
  Point2f a = triangle[pivot_idx] - triangle[(pivot_idx + 1) % 3];
  Point2f b = triangle[pivot_idx] - triangle[(pivot_idx + 2) % 3];
  float dot = a.dot(b);
  float angle = acos(dot / norm(a) / norm(b));
  return angle;
}

float angles(vector<Point2f> &p, vector<Point2f> &q, vector<perm_t> &perm) {
  perm_t pair0 = perm[0];
  perm_t pair1 = perm[1];
  perm_t pair2 = perm[2];

  float angle_p0 = getAngle(p, pair0.first);
  float angle_p1 = getAngle(p, pair1.first);
  float angle_p2 = getAngle(p, pair2.first);
  float angle_q0 = getAngle(q, pair0.second);
  float angle_q1 = getAngle(q, pair1.second);
  float angle_q2 = getAngle(q, pair2.second);

  float sines_diff_0 = abs(sin(angle_p0) - sin(angle_q0));
  float sines_diff_1 = abs(sin(angle_p1) - sin(angle_q1));
  float sines_diff_2 = abs(sin(angle_p2) - sin(angle_q2));

  float sinesMean = mean(sines_diff_0, sines_diff_1, sines_diff_2);
  return exp(-sinesMean / SIGMA);
}

float oppositeSide(vector<Point2f> &triangle, int pivot_idx) {
  float side = norm(triangle[(pivot_idx + 1) % 3] -
                    triangle[(pivot_idx + 2) % 3]);
  return side;
}

double ratios(vector<Point2f> &p, vector<Point2f> &q, vector<perm_t> &perm) {
  perm_t pair0 = perm[0];
  perm_t pair1 = perm[1];
  perm_t pair2 = perm[2];

  float R1 = oppositeSide(p, pair0.first) / oppositeSide(q, pair0.second);
  float R2 = oppositeSide(p, pair1.first) / oppositeSide(q, pair1.second);
  float R3 = oppositeSide(p, pair2.first) / oppositeSide(q, pair2.second);
  return exp(-stdDev(R1, R2, R3) / SIGMA);
}

double descriptors(vector<Mat> &dp, vector<Mat> &dq, vector<perm_t> &perm) {
  perm_t pair0 = perm[0];
  perm_t pair1 = perm[1];
  perm_t pair2 = perm[2];
  float desc_diff_0 = norm(dp[pair0.first] - dq[pair0.second]);
  float desc_diff_1 = norm(dp[pair1.first] - dq[pair1.second]);
  float desc_diff_2 = norm(dp[pair2.first] - dq[pair2.second]);
  float descMean = mean(desc_diff_0, desc_diff_1, desc_diff_2);
  return exp(- descMean / SIGMA);
}

namespace sim {
  MatchSimilarity similarity(vector<Point2f> &p, vector<Point2f> &q,
                                  vector<Mat> &dp, vector<Mat> &dq,
                                  float cang, float crat, float cdesc) {
    float s = cang + crat + cdesc;
    cang  /= s;
    crat  /= s;
    cdesc /= s;

    MatchSimilarity final_similarity;
    for (int i = 0; i < PERMS.size(); i++) {
      vector<perm_t> cur_perm = PERMS[i];
      float _angles_sim = angles(p, q, cur_perm);
      float _ratios_sim = ratios(p, q, cur_perm);
      float _desc_sim   = descriptors(dp, dq, cur_perm);
      float _total_sim = cang * _angles_sim + crat * _ratios_sim +
                         cdesc * _desc_sim;

      if (_total_sim > final_similarity.global_sim) {
        final_similarity.match_indices = cur_perm;
        final_similarity.global_sim = _total_sim;
        final_similarity.angles_sim = _angles_sim;
        final_similarity.ratios_sim = _ratios_sim;
        final_similarity.desc_sim = _desc_sim;
      }
    }
    return final_similarity;
  }
}
