/**
    main.cpp
    Purpose: Find visual correspondences between two sets of
    features from a pair of images using CUDA.

    @author Leiver Andres Campeón <leiverandres04p@hotmail.com>
    @author Yensy Helena Gomez <yensy@sirius.utp.edu.co>
    @author Juan Sebastián Vega Patiño <sebas060495@gmail.com>
    @author John Osorio <john@sirius.utp.edu.co>
    @version 1.0 29/12/16
*/

#include <iostream>
#include <cstdio>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include "match.hpp"
#include "draw.hpp"
#include <time.h>

using namespace cv;
using namespace std;
using namespace cv::gpu;



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

int vectorVectorToArray(vector<vector<int>> &edges, int *array){
    int i;
    for(i=0; i<edges.size(); i++){
        array[i*3+0] = edges[i][0];
        array[i*3+1] = edges[i][1];
        array[i*3+2] = edges[i][2];

    }
    return 0;
}

int keyPointsToArray(vector<KeyPoint> kpts, float *array){
    for (int i = 0; i < kpts.size(); i++) {
        array[i*2+0] = kpts[i].pt.x;
        array[i*2+1] = kpts[i].pt.y;
    }
    return 0;
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

  //draw::triangulation(img, triangleList);

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

  clock_t begin = clock();
  GpuMat d_img1( imread("./test-images/monster1s.JPG", 0) );
  GpuMat d_img2( imread("./test-images/monster1m.JPG", 0) );


  Mat img1 = imread("./test-images/monster1s.JPG",0);
  Mat img2 = imread("./test-images/monster1m.JPG",0);

  // For Surf detection
  int minHessian = 400;

  SURF_GPU surf(minHessian);
  GpuMat d_kpts1, d_kpts2;
  GpuMat d_descriptor1, d_descriptor2;
  surf(d_img1, GpuMat(), d_kpts1, d_descriptor1);
  surf(d_img2, GpuMat(), d_kpts2, d_descriptor2);

  vector<KeyPoint> kpts1, kpts2;
  Mat descriptor1, descriptor2;

  surf.downloadKeypoints(d_kpts1, kpts1);
  surf.downloadKeypoints(d_kpts2, kpts2);

  d_descriptor1.download(descriptor1);
  d_descriptor2.download(descriptor2);
  clock_t end = clock();
  double time_spent = (double)(end-begin) / CLOCKS_PER_SEC;

  cout << endl << "Surf execution time: " << time_spent << "s" << endl;

  cout << endl << kpts1.size() << " Keypoints Detected in image 1 " << endl;
  cout << endl << kpts2.size() << " Keypoints Detected in image 2 " << endl;

  // Building hyperedges Matrices
  cout << endl << "Triangulating ..." << endl;
  vector<vector<int> > Edges1 = delaunayTriangulation(img1, kpts1);
  vector<vector<int> > Edges2 = delaunayTriangulation(img2, kpts2);

  // Conversion to c array
  int *edges1Array = (int*)malloc(3*Edges1.size()*sizeof(int));
  int *edges2Array = (int*)malloc(3*Edges2.size()*sizeof(int));
  vectorVectorToArray(Edges1, edges1Array);
  cout << edges1Array[0*3+0] << " " << edges1Array[0*3+1] << "Test Edges" << endl;
  cout << Edges1[0][0] << " " << Edges1[0][1] << "Reales" << endl;


  // Conversion of KeyPoint to array
  // La columna cero del arreglo hace referencia a X y la 1 a Y
  float *keyPoints1Array, *keyPoints2Array;
  keyPoints1Array = (float*)malloc(kpts1.size()*sizeof(float)*2);
  keyPoints2Array = (float*)malloc(kpts2.size()*sizeof(float)*2);
  keyPointsToArray(kpts1, keyPoints1Array);
  keyPointsToArray(kpts2, keyPoints2Array);
  cout << keyPoints1Array[0*2+0] << " " << keyPoints1Array[0*2+1] << "Test Keypoints" << endl;
  cout << kpts1[0].pt.x << " " << kpts1[0].pt.y << "KeyPoints Reales" << endl;

  cout << Edges1.size() << " Edges from image 1" << endl;
  cout << Edges2.size() << " Edges from image 2" << endl;
  cout << endl << "Matching ..." << endl;

  vector<pair<int, int> > edge_matches = match::hyperedges(Edges1, Edges2,
                                                           kpts1,
                                                           kpts2,
                                                           descriptor1,
                                                           descriptor2, 10, 10,
                                                           3, 0.75);

  cout << endl << "Edges Matching done. ";
  cout << edge_matches.size() << " edge matches passed!" << endl;

  vector<DMatch> matches = match::points(edge_matches, descriptor1, descriptor2,
                                         Edges1, Edges2, 0.1);

  cout << endl << "Point Matching Done. ";
  cout << matches.size() << " Point matches passed!" << endl;

  // Draw Point matching
  //draw::pointsMatch(img1, kpts1, img2, kpts2, matches);

  free(edges1Array); free(keyPoints1Array); free(keyPoints2Array);
  return 0;
}
