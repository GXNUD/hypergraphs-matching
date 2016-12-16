#include "iostream"
#include "stdio.h"
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// Distancia ecluidinana entre los puntos de una misma imagen usando
// descriptores
Mat euclidDistance(Mat &vect1) {
  Mat matrixEuclidian(vect1.rows, vect1.rows, DataType<double>::type);
  for (int i = 0; i < vect1.rows; i++){
    for (int j = 0; j < vect1.rows; j++){
      double sum = 0.0;
      for (int k = 0; k < vect1.cols; k++){
        double e_i = vect1.at<double>(k, i);
        double e_j = vect1.at<double>(k, j);
        cout << e_i << " " << e_j << " : " << ((e_i - e_j) * (e_i - e_j)) << endl;
        sum += ((e_i - e_j) * (e_i - e_j));
      }
      matrixEuclidian.at<double>(i,j) = sqrt(sum);
    }
  }
  cout << "Distancias en la función" << endl;
  for (int i = 0; i < vect1.rows; i++) {
    for (int j = 0; j < vect1.rows; j++) {
      cout << matrixEuclidian.at<double>(i, j) << " ";
    }
    cout << endl;
  }
  return matrixEuclidian;
}

// //Distancias entre los puntos caracteristicas de una imagen1 usando la
// posición del punto.
Mat distancePoints(vector<KeyPoint> &point) {
  int n = point.size();
  Mat matrixEuclidian(n, n, DataType<float>::type),
      matrixEuclidianSort(n, n, DataType<int>::type);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float x1 = point[i].pt.x;
      float x2 = point[j].pt.x;
      float y1 = point[i].pt.y;
      float y2 = point[j].pt.y;
      matrixEuclidian.at<float>(i, j) =
          sqrt(((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)));
    }
  }
  // cv::sortIdx(matrixEuclidian, matrixEuclidianSort , CV_SORT_ASCENDING);

  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < n; j++) {
  //     cout << matrixEuclidian.at<float>(i, j) << " ";
  //   }
  //   cout << endl;
  // }
  // cout<<"ordenado" <<endl;
  // for (int i = 0; i < 10; i++) {
  //   for (int j = 0; j < 10; j++) {
  //     cout << matrixEuclidianSort.at<int>(i,j) << " ";
  //   }
  //   cout << endl;
  // }
  return matrixEuclidian;
}

// distancia del arcoseno.
Mat distanceBetweenImg(Mat &vec1, Mat &vec2) {
  Mat similarity(vec1.rows, vec2.rows, DataType<float>::type);
  for (int i = 0; i < vec1.rows; i++) {
    for (int j = 0; j < vec2.rows; j++) {
      float producto = 0.0;
      float norma1 = 0.0;
      float norma2 = 0.0;
      for (int k = 0; k < vec1.cols; k++) {
        norma1 += vec1.at<float>(k, i) * vec1.at<float>(k, i);
        norma2 += vec2.at<float>(k, j) * vec2.at<float>(k, j);
        producto += (vec1.at<float>(k, i) * vec2.at<float>(k, j));
      }
      similarity.at<float>(i, j) = (producto / ((sqrt(norma1) * sqrt(norma2))));
    }
  }

  // for (int i = 0; i < 10; i++) {
  //   for (int j = 0; j < 10; j++) {
  //     if (similarity.at<double>(i,j)> 0.5000000) {
  //       cout << similarity.at<float>(j, i) << " ";
  //     }
  //   }
  //   cout << endl;
  // }
  return similarity;
}

/*Algoritmo de los k vecinos más cercanos, esto nos permitira conocer
los indices de la matrix de los vecinos más cercanos,
para hacer el hipergrafo de la img1 como de img2 */
Mat KNN(Mat &matEucl) {
  Mat indices(matEucl.rows, 3, DataType<int>::type);
  float minDist = 10e6;
  int minIdx1 = -1;
  int minIdx2 = -1;
  for (int i = 0; i < matEucl.rows; i++) {
    for (int j = 0; j < matEucl.cols; j++) {
      if ((matEucl.at<float>(i, j) <= minDist) && (j != i)) {

        minDist = matEucl.at<float>(i, j);
        minIdx1 = j;
      }
    }
    minDist = 1e6;
    for (int j = 0; j < matEucl.cols; j++) {
      if ((matEucl.at<float>(i, j) <= minDist) && (j != minIdx1) && (j != i)) {
        minDist = matEucl.at<float>(i, j);
        minIdx2 = j;
      }
      // como guardar los indices en una Matriz de N por 3;
    }
    cout << "Indice0: " << i << " "
         << "indice1: " << minIdx1 << " "
         << "indice2: " << minIdx2 << endl;
    indices.at<int>(i, 0) = i;
    indices.at<int>(i, 1) = minIdx1;
    indices.at<int>(i, 2) = minIdx2;
  }
  return indices;
}

/*
########  ########     ###    ##      ## #### ##    ##  ######
##     ## ##     ##   ## ##   ##  ##  ##  ##  ###   ## ##    ##
##     ## ##     ##  ##   ##  ##  ##  ##  ##  ####  ## ##
##     ## ########  ##     ## ##  ##  ##  ##  ## ## ## ##   ####
##     ## ##   ##   ######### ##  ##  ##  ##  ##  #### ##    ##
##     ## ##    ##  ##     ## ##  ##  ##  ##  ##   ### ##    ##
########  ##     ## ##     ##  ###  ###  #### ##    ##  ######
*/

void drawDelaunay(Mat &img, vector<Vec6f> &triangleList) {
  Scalar delaunay_color(255,255,255);
  vector<Point> pt(3);
  Size size = img.size();
  Rect rect(0, 0, size.width, size.height);
  int count_outliers = 0;
  for( size_t i = 0; i < triangleList.size(); i++ ) {
    Vec6f t = triangleList[i];
    pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
    pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
    pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

    // Draw rectangles completely inside the image.
    if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
      line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
      line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
      line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
    } else {
      count_outliers++;
    }
  }
  cout << "[drawDelaunay] " <<count_outliers << " points are not in rect" << endl;
  imshow("Delaunay Triangulation", img);
  waitKey(0);
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

vector< vector<int> > delaunayTriangulation(Mat img, vector<KeyPoint> kpts) {
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
  Rect rect(0, 0, size.width, size.height); // TODO
  Subdiv2D subdiv(rect);
  subdiv.insert(points);
  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);

  drawDelaunay(img, triangleList);

  // Converting to edges with coordinates to indices
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
  
  cout << "[delaunayTriangulation] ";
  cout << "Size of input KeyPoints " << kpts.size() << endl;
  cout << "[delaunayTriangulation] ";
  cout << "Size of input points " << points.size() << endl;
  cout << "[delaunayTriangulation] ";
  cout << "Size of trinagleList: " << triangleList.size() << endl;
  cout << "[delaunayTriangulation] ";
  cout << "Number of Rect outliers " << rect_count_outliers << endl;
  cout << "[delaunayTriangulation] ";
  cout << "Number of Map outliers " << map_count_outliers << endl;
  cout << "[delaunayTriangulation] ";
  int outliers = rect_count_outliers + map_count_outliers;
  if ((triangleList.size() - outliers) != edges.size()) {
    cout << "Edges has more elements than it should" << endl;
  } {
    cout << "Edges has exactly (trinagleList.size() - outliers) elements" << endl;
  }
  // for (int i = 0; i < edges.size(); i++) {
  //   cout << "Edges number " << i << " : " << endl;
  //   cout << edges[i][0] << " ";
  //   cout << edges[i][1] << " ";
  //   cout << edges[i][2] << endl;
  // }

  return edges;
}


void positionXYIJK(Mat &indice, vector<KeyPoint> &point){
  float size = indice.rows*sizeof(float);
  float  *determinant;
  determinant = (float *) malloc(size);

  for (int i = 0; i < indice.rows; i++) {
      float x1 = point[indice.at<int>(i, 0)].pt.x;
      float y1 = point[indice.at<int>(i, 0)].pt.y;
      float x2 = point[indice.at<int>(i, 1)].pt.x;
      float y2 = point[indice.at<int>(i, 1)].pt.y;
      float x3 = point[indice.at<int>(i, 2)].pt.x;
      float y3 = point[indice.at<int>(i, 2)].pt.y;
      determinant[i] = (x1-x3)*(y2-y3)-(x2-x3)*(y1-y3);
      cout << "V1: " << x1 << ", " << y1 << endl;
      cout << "V2: " << x2 << ", " << y2 << endl;
      cout << "V3: " << x3 << ", " << y3 << endl;
      cout << determinant[i] << endl;

    }
  free(determinant);
}

int main(int argc, const char *argv[]) {
  const Mat img1 = imread("./house/house.seq0.png", 0); // Load as grayscale
  const Mat img2 = imread("./house/house.seq0.png", 0); // Load as grayscale

  SurfFeatureDetector detector(10);

  vector<KeyPoint> keypoints1;
  vector<KeyPoint> keypoints2;
  detector.detect(img1, keypoints1);
  detector.detect(img2, keypoints2);
  Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create("SIFT");

  Mat descriptor1, descriptor2;

  descriptor->compute(img1, keypoints1, descriptor1);
  descriptor->compute(img2, keypoints2, descriptor2);

  // Add results to image and save.
  Mat output1;
  Mat output2;

  drawKeypoints(img1, keypoints1, output1);
  imwrite("sift_result.jpg", output1);
  drawKeypoints(img2, keypoints2, output2);
  imwrite("sift_result1.jpg", output2);

  Mat dist1 = distancePoints(keypoints1);
  Mat dist2 = distancePoints(keypoints2);

  // build hyperedges vector
  vector<vector<int> > Edges1 = delaunayTriangulation(img1, keypoints1);
  cout << "Edges1 has: " << Edges1.size() << " elements" << endl;
  // Mat Edges1 = KNN(dist1);
  // Mat Edges2 = KNN(dist2);

  // edge matching
  // Mat matches = matchHyperedges(Edges1, Edges2, keypoints1, keypoints2);
  // point matching

  // draw matching

  // Mat matSimilarity = distanceBetweenImg(descriptor1, descriptor2);
  // for (int i = 0; i < matSimilarity.rows; i++) {
  //   for (int j = 0; j < matSimilarity.cols; j++) {
  //     cout << "Similarity " << i << " and " << j << ": " << matSimilarity.at<double>(i, j) << endl;
  //   }
  // }
  // Mat dist1 = euclidDistance(descriptor1);
  // for (int i = 0; i < dist1.rows; i++) {
  //   for (int j = 0; j < dist1.cols; j++) {
  //     cout << dist1.at<double>(i, j) << " ";
  //   }
  //   cout << endl;
  // }
  // euclidDistance(descriptor2);
  // KNN(prueba);
  // Mat prueba2(keypoints.size(), 3, DataType<float>::type);
  // positionXYIJK(prueba2,keypoints);

  // for (int i = 0; i < keypoints.size(); i++) {
  //   cout << "DescriptorA (" << descriptor1.row(i) << ")" << endl;
  // }

  return 0;
}
