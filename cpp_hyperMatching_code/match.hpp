#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "similarity.hpp"

using namespace std;
using namespace cv;

struct hyperedgeMatch {
  int p_idx; // Point from first image matched
  int q_idx; // Point from second image matched
  float total_sim;
  float angles_sim;
  float ratios_sim;
  float desc_sim;
};

namespace match {
  pair<vector<hyperedgeMatch>, vector<DMatch>> match(
    vector<vector<int> > &edges1, vector<vector<int> > &edges2,
    vector<KeyPoint> &kpts1, vector<KeyPoint> &kpts2,
    Mat &desc1, Mat &desc2, float cang, float cratm, float cdesc,
    float th_edges, float th_points
  ) {
    vector<hyperedgeMatch> hyperedge_matches;
    vector<DMatch> point_matches;
    set<pair<int, int>> selected_point_matches;

    for (int i = 0; i < edges1.size(); i++) {
      float max_similarity = -1;
      int best_index = -1;
      float s_ang = -1, s_ratios = -1, s_desc = -1;
      vector<pair<int, int>> match_kpts_indices(3);
      for (int j = 0; j < edges2.size(); j++) {
        vector<Point2f> p_points(3), q_points(3);
        vector<Mat> desc_p(3), desc_q(3);
        for (int k = 0; k < 3; k++) {
          p_points[k] = kpts1[edges1[i][k]].pt;
          q_points[k] = kpts2[edges2[j][k]].pt;
          desc_p[k] = desc1.row(edges1[i][k]);
          desc_p[k] = desc2.row(edges2[j][k]);
        }

        auto t = sim::similarity(p_points, q_points, desc_p, desc_q,
                            cang, cratm, cdesc);
        vector<perm_t> match_indices = t.match_indices;
        float sim_global = t.global_sim;
        float sim_a = t.angles_sim;
        float sim_r = t.ratios_sim;
        float sim_d = t.desc_sim;

        if (sim_global > max_similarity) {
          best_index = j;
          max_similarity = sim_global;
          s_ang = sim_a;
          s_ratios = sim_r;
          s_desc = sim_d;
          for (int l = 0; l < match_indices.size(); l++) {
            int p_i = edges1[i][match_indices[l].first];
            int q_i = edges2[j][match_indices[l].second];
            match_kpts_indices[l] = make_pair(p_i, q_i);
          }
        }
      }

      if (max_similarity >= th_edges) {
        hyperedgeMatch cur_match;
        cur_match.p_idx = i;
        cur_match.q_idx = best_index;
        cur_match.total_sim = max_similarity;
        cur_match.angles_sim = s_ang;
        cur_match.ratios_sim = s_ratios;
        cur_match.desc_sim = s_desc;

        hyperedge_matches.push_back(cur_match);
        for (int m = 0; m < match_kpts_indices.size(); m++) {
          int idx1_m = match_kpts_indices[m].first;
          int idx2_m = match_kpts_indices[m].second;
          float dist = norm(desc1.row(idx1_m) - desc2.row(idx2_m));
          float points_sim = exp(-dist / SIGMA);
          if (selected_point_matches.count(match_kpts_indices[m]) == 0 &&
              points_sim >= th_points) {
            point_matches.push_back(DMatch(idx1_m, idx2_m, dist));
            selected_point_matches.insert(match_kpts_indices[m]);
          }
        }
      }
    }
    pair<vector<hyperedgeMatch>, vector<DMatch>> edge_and_points_matches;
    edge_and_points_matches = make_pair(hyperedge_matches, point_matches);
    return edge_and_points_matches;
  }
}
