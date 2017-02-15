/**
    main.cpp
    Purpose: Find visual correspondences between two sets of
    features from a pair of images

    @author Leiver Andres Campeón <leiverandres04p@hotmail.com>
    @author Yensy Helena Gomez <yensy@sirius.utp.edu.co>
    @author Juan Sebastián Vega Patiño <sebas060495@gmail.com>
    @version 1.0 29/12/16
*/

#include <iostream>
#include <cstdio>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "match.hpp"
#include "draw.hpp"

using namespace cv;
using namespace std;


/*
##     ## ######## #### ##        ######
##     ##    ##     ##  ##       ##    ##
##     ##    ##     ##  ##       ##
##     ##    ##     ##  ##        ######
##     ##    ##     ##  ##             ##
##     ##    ##     ##  ##       ##    ##
 #######     ##    #### ########  ######
*/

/**
  Sums up elements of a vector

  @param vec vector containing elements to accumulate
  @return sum of elements
*/
long double accum(vector<double> &vec) {
  long double sum = 0.0;
  vector<double>::iterator it;
  for (it = vec.begin(); it != vec.end(); it++) {
    sum += *it;
  }
  return sum;
}

template<typename T>
vector<vector<T> > getPermutation(vector<T> data) {
  vector<vector<T> > perms;
  sort(data.begin(), data.end());
  do {
    perms.push_back(data);
  } while(next_permutation(data.begin(), data.end()));
  return perms;
}

bool responseCMP(const KeyPoint& p1, const KeyPoint& p2) {
    return p1.response > p2.response;
}

/*
######## ########   ######   ########  ######
##       ##     ## ##    ##  ##       ##    ##
##       ##     ## ##        ##       ##
######   ##     ## ##   #### ######    ######
##       ##     ## ##    ##  ##             ##
##       ##     ## ##    ##  ##       ##    ##
######## ########   ######   ########  ######
*/

/**
  Obtain a list of hyperedges from the Delaunay Triangulation computed with
  some Image Keypoints

  @param img Image from which keypoints are extracted
  @param kpts
  @return
*/

vector<vector<int> > delaunayTriangulation(Mat img, vector<KeyPoint> kpts) {
  vector<Point2f> points;
  KeyPoint::convert(kpts, points);
  map<pair<double, double> , int> pt_idx;

  // Mapping points with their indices
  for (int i = 0; i < points.size(); i++) {
    Point2f p = points[i];
    pt_idx[make_pair(p.x, p.y)] = i;
  }

  // Triangulation
  Size size = img.size();
  Rect rect(0, 0, size.width, size.height);
  Subdiv2D subdiv(rect);
  subdiv.insert(points);
  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);

  draw::triangulation(img, triangleList);

  // Converting to edges from coordinates to indices
  int rect_count_outliers = 0;
  int map_count_outliers = 0;
  vector<Point2f> pt(3);
  vector<vector<int> > edges;
  for (int i = 0; i < triangleList.size(); i++) {
    Vec6f t = triangleList[i];
    pt[0] = Point2f(t[0], t[1]);
    pt[1] = Point2f(t[2], t[3]);
    pt[2] = Point2f(t[4], t[5]);
    if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
      pair<double, double> p0 = make_pair(pt[0].x, pt[0].y);
      pair<double, double> p1 = make_pair(pt[1].x, pt[1].y);
      pair<double, double> p2 = make_pair(pt[2].x, pt[2].y);
      if (pt_idx.count(p0) && pt_idx.count(p1) && pt_idx.count(p2)) {
        vector<int> edge(3);
        edge[0] = pt_idx[p0];
        edge[1] = pt_idx[p1];
        edge[2] = pt_idx[p2];
        edges.push_back(edge);
      } else {
        map_count_outliers++;
      }
    } else {
      rect_count_outliers++;
    }
  }

  return edges;
}

/*
##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##
*/

int main(int argc, const char *argv[]) {
  Mat img1 = imread("./house/house.seq0.png", 0);
  Mat img2 = imread("./house/house.seq0.trans.png", 0);
  Mat out_img;

  // For Surf detection
  int minHessian = 400;

  SurfFeatureDetector detector(minHessian);
  // SiftFeatureDetector detector(0.05, 5.0);
  vector<KeyPoint> kpts1, kpts2;
  detector.detect(img1, kpts1);
  detector.detect(img2, kpts2);
  sort(kpts1.begin(), kpts1.end(), responseCMP);
  sort(kpts2.begin(), kpts2.end(), responseCMP);

  // Test vectors with less points
  int limit = 20;
  vector<KeyPoint> t_kpts1(kpts1.begin(), kpts1.begin() + limit);
  vector<KeyPoint> t_kpts2(kpts2.begin(), kpts2.begin() + limit);

  SurfDescriptorExtractor extractor;
  // SiftDescriptorExtractor extractor;
  Mat descriptor1, descriptor2;
  extractor.compute(img1, t_kpts1, descriptor1);
  extractor.compute(img2, t_kpts2, descriptor2);

  // Add results to an image and save them.
  Mat output1;
  Mat output2;

  drawKeypoints(img1, t_kpts1, output1);
  imwrite("surf_result1.jpg", output1);
  drawKeypoints(img2, t_kpts2, output2);
  imwrite("surf_result2.jpg", output2);

  // Building hyperedges Matrices
  vector<vector<int> > Edges1 = delaunayTriangulation(img1, t_kpts1);
  vector<vector<int> > Edges2 = delaunayTriangulation(img2, t_kpts2);

  cout << "Matching ..." << endl;

  vector<pair<int, int> > edge_matches = match::hyperedges(Edges1, Edges2,
                                                           t_kpts1,
                                                           t_kpts2,
                                                           descriptor1,
                                                           descriptor2, 5, 10,
                                                           5, 0.85);


  vector<DMatch> matches = match::points(edge_matches, descriptor1, descriptor2,
                                         Edges1, Edges2, 0.1);

  cout << "Matching Done!" << endl;

  // Draw Edges matching
  // draw::edgesMatch(
  //   img1, img2, edge_matches, Edges1, Edges2, t_kpts1, t_kpts2
  // );

  // Draw Point matching
  drawMatches(img1, t_kpts1, img2, t_kpts2, matches, out_img);
  imshow("Matches", out_img);
  waitKey(0);
  return 0;
}
