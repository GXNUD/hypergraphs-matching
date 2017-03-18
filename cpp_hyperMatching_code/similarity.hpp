#include <vector>
#include <cmath>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "combination.hpp"

using namespace cv;
using namespace std;

double vectorsAngleSin(Point2f &pivot, Point2f &p, Point2f &q) {
  Point2f v1 = p - pivot;
  Point2f v2 = q - pivot;
  double dot = v1.dot(v2);
  double angle = acos(dot / norm(v1) / norm(v2));
  return sin(angle);
}

vector<double> getAnglesSin(vector<Point2f> &p) {
  Point2f &p1 = p[0];
  Point2f &p2 = p[1];
  Point2f &p3 = p[2];
  vector<double> sines(3);
  sines[0] = vectorsAngleSin(p1, p2, p3);
  sines[1] = vectorsAngleSin(p2, p1, p3);
  sines[2] = vectorsAngleSin(p3, p1, p2);
  return sines;
}

bool compMat(Mat &a, Mat &b) {
  return norm(a) < norm(b);
}

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
}

typedef pair<int,int> perm_t;
vector<vector<pair<int, int>>> PERMS = {
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)},
  {perm_t(0, 0), perm_t(1, 1), perm_t(2, 2)}
};

struct MatchSimilarity {
  vector<perm_t> indices_matched; // pairs of indices of best configuration between the hyperedges
  double global_sim;
  double angles_sim;
  double ratios_sim;
  double desc_sim;
};

namespace sim {
  double angles(vector<Point2f> &p, vector<Point2f> &q, double sigma = 0.5) {
      vector<double> sines1 = getAnglesSin(p);
      vector<double> sines2 = getAnglesSin(q);

      sort(sines1.begin(), sines1.end());
      double min_diff_between_sin = 1E30;
      do {
          double sum = 0;
          for (int i = 0; i < 3; i++)
              sum += fabs(sines1[i] - sines2[i]);
          min_diff_between_sin = min(min_diff_between_sin, sum);
      } while (next_permutation(sines1.begin(), sines1.end()));

      return exp(-min_diff_between_sin / sigma);
  }

  float oppositeSide(vector<Point2f> &points, int pivot) {
    float side = norm(p[(i + 1) % 3] - p[(i + 2) %3]);
    return side;
  }

  double ratios(vector<Point2f> &p, vector<Point2f> &q, vector<perm_t> &perm) {
   Point2f pair0 = perm[0];
   Point2f pair1 = perm[1];
   Point2f pair2 = perm[2];
   float R1 = oppositeSide(p, pair0.x) / oppositeSide(q, pair0.y);
   float R2 = oppositeSide(p, pair1.x) / oppositeSide(q, pair1.y);
   float R3 = oppositeSide(p, pair2.x) / oppositeSide(q, pair2.y);
   return exp(-stdDev(R1, R2, R3) / SIGMA);
  }

  double descriptors(vector<Mat> &desc1, vector<Mat> &desc2, double sigma = 0.5) {
      double min_diff = 1E30;
      sort(desc2.begin(), desc2.end(), compMat);
      do {
          double diff = 0;
          for (int i = 0; i < 3; i++) {
              diff += norm(desc2[i] - desc1[i]);
          }
          min_diff = min(diff, min_diff);
      } while (next_permutation(desc2.begin(), desc2.end(), compMat));
      return exp(- min_diff / sigma);
  }

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

      if (_total_sim > max_sim) {
        final_similarity.global_sim = _total_sim;
        final_similarity.indices_matched = cur_perm;
        final_similarity.angles_sim = _angles_sim;
        final_similarity.ratios_sim = _ratios_sim;
        final_similarity.desc_sim = _desc_sim;
      }
    }
    return final_similarity;
  }
}
