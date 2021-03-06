#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "similarity.hpp"

using namespace std;
using namespace cv;

namespace match {
    vector< pair<int, int> > hyperedges(vector<vector<int> > &edges1,
                                        vector<vector<int> > &edges2,
                                        vector<KeyPoint> &kp1,
                                        vector<KeyPoint> &kp2,
                                        Mat &desc1, Mat &desc2,
                                        double cang, double crat, double cdesc,
                                        double thresholding) {
        double sigma = 0.5;
        vector< pair<int, int> > matches;
        double _sum = cang + crat + cdesc;
        cang /= _sum;
        crat /= _sum;
        cdesc /= _sum;

        for (size_t i = 0; i < edges1.size(); i++) {
            int best_match_idx = -1;
            double max_similarity = -1E30;

            for (size_t j = 0; j < edges2.size(); j++) {
                vector<Point2f> e1_points(3), e2_points(3);
                vector<Mat> des_1(3), des_2(3);
                for (int k = 0; k < 3; k++) {
                    e1_points[k] = kp1[edges1[i][k]].pt;
                    e2_points[k] = kp2[edges2[j][k]].pt;
                    des_1[k] = desc1.row(edges1[i][k]);
                    des_2[k] = desc2.row(edges2[j][k]);
                }
                double sim_angles = sim::angles(e1_points, e2_points, sigma);
                double sim_ratios = sim::ratios(e1_points, e2_points, sigma);
                double sim_desc = sim::descriptors(des_1, des_2, sigma);
                double similarity = cang * sim_angles + crat * sim_ratios +
                                    cdesc * sim_desc;

                if (similarity > max_similarity) {
                    best_match_idx = j;
                    max_similarity = similarity;
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

    vector<DMatch> points(
        vector<pair<int, int> > edge_matches,
        Mat &desc1, Mat &desc2,
        vector<vector<int> > &edges1, vector<vector<int> > &edges2,
        double th, double sigma = 0.5
    ) {
        vector<DMatch> matches;
        set<pair<int, int> > S;
        for (size_t i = 0; i < edge_matches.size(); i++) {
            int base_edge_idx = edge_matches[i].first;
            int ref_edge_idx  = edge_matches[i].second;
            vector<Mat> des_1(3), des_2(3);
            for (int k = 0; k < 3; k++) {
                des_1[k] = desc1.row(edges1[base_edge_idx][k]);
                des_2[k] = desc2.row(edges2[ref_edge_idx][k]);
            }

            vector<int> best_match(3);
            vector<double> best_sim(3, -1E30);
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    double _sim = exp(-sum(abs(des_1[j] - des_2[k]))[0] / sigma);

                    if (_sim > best_sim[j]) {
                        best_sim[j] = _sim;
                        best_match[j] = k;
                    }
                }
            }

            for (int j = 0; j < 3; j++) {
                int k = best_match[j];

                int qI = edges1[base_edge_idx][j];
                int tI = edges2[ref_edge_idx][k];

                if (!S.count(make_pair(qI, tI)) && best_sim[j] > th) {
                    double _dist = -sigma * log(best_sim[j]);
                    matches.push_back(DMatch(qI, tI, _dist));
                    S.insert(make_pair(qI, tI));
                }
            }
        }
        return matches;
    }
}
