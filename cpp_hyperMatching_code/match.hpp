#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "similarity.hpp"

using namespace std;
using namespace cv;

const double SIGMA = 0.5;

namespace match {
    
  pair<vector<pair<int,int>>, vector<DMatch>> match(
    vector<vector<int> > &edges1, vector<vector<int> > &edges2,
    vector<KeyPoint> &kpts1, vector<KeyPoint> &kpts2,
    Mat &desc1, Mat &desc2, float cang, float cratm, float cdesc,
    int th_edges, int th_points
  ) {
    vector<pair<int, int>> hyperedge_matches;
    vector<DMatch> point_matches;
    set<pair<int, int>> selected_point_matches;

    for (int i = 0; i < edges1.size(); i++) {
      double max_similarity = -1;
      int best_index = -1;
      double s_ang = -1, s_ratios = -1, s_desc -1;
      vector<pair<int, int>> edge_match_indices(3);
      for (int j = 0; j < edges2.size(); j++) {
        vector<Point2f> p_points(3), q_points(3);
        vector<Mat> desc_p(3), desc_q(3);
        for (int k = 0; k < 3; k++) {
          p_points[k] = kpts1[edges1[i][k]].pt;
          q_points[k] = kpts2[edges2[j][k]].pt;
          desc_p[k] = desc1.row(edges1[i][k]);
          desc_p[k] = desc2.row(edges2[j][k]);
        }

        auto t = similarity(p_points, q_points, desc_p, desc_q
                                cang, cratm, cdesc);
        vector<pair<int,int>> point_idx = get<0>(t);
        double sim_global = get<1>(t);
        double sim_a = get<2>(t);
        double sim_r = get<3>(t);
        double sim_d = get<4>(t);

        if (sim_global > max_similarity) {
          best_index = j;
          max_similarity = sim_global;
          s_ang = sim_a;
          s_ratios = sim_r;
          s_desc = sim_d;
          for (int l = 0; l < point_idx.size(); l++) {
            int p_i = point_idx[l].first;
            int q_i = point_idx[l].second;
            edge_match_indices[l] = make_pair(p_i, q_i);
          }
        }
      }

      if (max_similarity >= th_edges) {
        auto edge_match = make_tuple(
          i, best_index, max_similarity, s_ang, s_ratios, s_desc
        );
        hyperedge_matches.push_back(edge_match);
        for (int m = 0; m < edge_match_indices.size(); m++) {
          int idx1_m = edge_match_indices[m].first;
          int idx2_m = edge_match_indices[m].second;
          double dist = norm(desc1[idx1_m] - desc2[idx2_m]);
          double points_sim = exp(-dist / SIGMA)
          if (!selected_point_matches.count(edge_match_indices[m]) && points_sim >= th_points) {
            point_matches.push_back(DMatch(idx1_m, idx2_m, dist));
            selected_point_matches.insert(edge_match_indices[m]);
          }
        }
      }
    }
  }
}
