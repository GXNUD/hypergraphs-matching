#include <vector>
#include <cmath>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "similarity"

using namespace std;
using namespace cv;

namespace match {
    vector< pair<int, int> > hyperedges(vector<vector<int> > &edges1,
                                        vector<vector<int> > &edges2,
                                        vector<KeyPoint> &kp1,
                                        vector<KeyPoint> &kp2,
                                        Mat &desc1, Mat &desc2,
                                        double c1, double c2, double c3,
                                        double thresholding) {
        double sigma = 0.5;
        vector< pair<int, int> > matches;
        double suma = c1 + c2 + c3;
        c1 /= suma;
        c2 /= suma;
        c3 /= suma;

        for (int i = 0; i < edges1.size(); i++) {
            int best_match_idx = -1;
            double max_similarity = -1E30;
            double s_ang = -1E30;
            double s_ratio = -1E30;
            double s_desc = -1E30;
            for (int j = 0; j < edges2.size(); j++) {
                vector<Point2f> e1_points(3), e2_points(3);
                vector<Mat> des_1(3), des_2(3);
                for (int k = 0; k < 3; k++) {
                    e1_points[k] = kp1[edges1[i][k]].pt;
                    e2_points[k] = kp2[edges2[j][k]].pt;
                    des_1[k] = desc1.row(edges1[i][k]);
                    des_2[k] = desc2.row(edges2[j][k]);
                }
                double sim_angles = sim::angles(e1_points, e2_points);
                double sim_ratios = sim::ratios(e1_points, e2_points);
                double sim_desc = descSimilarity(des_1, des_2);
                double similarity = c1 * sim_ratios + c2 * sim_angles +
                                    c3 * sim_desc;

                if (similarity > max_similarity) {
                    best_match_idx = j;
                    max_similarity = similarity;
                    s_area = area_dist;
                    s_ang = sim_angles;
                    s_desc = sim_desc;
                }
            }
            if (max_similarity >= thresholding) {
                matches.push_back(make_pair(i, best_match_idx));
            }
        }
        return matches;
    }

    double descDistance(Mat e1, Mat e2) {
      Mat diffs;
      absdiff(e1, e2, diffs);
      double dist = sum(diffs)[0];
      return dist;
    }

    vector<DMatch> matchPoints(vector<pair<int, int> > edge_matches, Mat &desc1,
                               Mat &desc2, vector<vector<int> > &edges1,
                               vector<vector<int> > &edges2, double th) {
      vector<DMatch> matches;
      set<pair<int, int> > S;
      for (int i = 0; i < edge_matches.size(); i++) {
        int base_edge_idx = edge_matches[i].first;
        int ref_edge_idx  = edge_matches[i].second;
        vector<Mat> des_1(3), des_2(3);
        for (int k = 0; k < 3; k++) {
            des_1[k] = desc1.row(edges1[base_edge_idx][k]);
            des_2[k] = desc2.row(edges2[ref_edge_idx][k]);
        }
        double max_sum = -1E30;
        int max_idx = -1;
        for (int j = 0; j < 3; j++) {
            double suma = 0;
            for (int k = 0; k < 3; k++) {
                suma = sum(abs(des2[k] - des_1[j]));
            }
            max_sum = max(suma, max_sum);
            if (suma > max_sum) {
                max_sum = suma;
                // TODO max_idx = ;
            }
        }
        sim_desc = exp(- suma / sigma);
        }
      }

      return matches;
    }
}
