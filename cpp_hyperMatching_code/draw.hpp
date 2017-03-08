#include <iostream>
#include <cstdio>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

namespace draw {
    void triangulation(Mat &img, vector<Vec6f> &triangleList) {
      Mat img_out;
      img.copyTo(img_out);
      Scalar delaunay_color(255,255,255);
      vector<Point> pt(3);
      Size size = img_out.size();
      Rect rect(0, 0, size.width, size.height);
      int count_outliers = 0;
      for( size_t i = 0; i < triangleList.size(); i++ ) {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw rectangles completely inside the image.
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
          line(img_out, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
          line(img_out, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
          line(img_out, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        } else {
          count_outliers++;
        }
      }
      // cout << "[drawDelaunay] " <<count_outliers << " points are not in rect" << endl;
      namedWindow("Delaunay Triangulation", WINDOW_NORMAL);
      resizeWindow("Delaunay Triangulation", 800, 900);
      imshow("Delaunay Triangulation", img_out);
      waitKey(0);
    }

    void edgesMatch(Mat &img1, Mat &img2, vector< pair<int, int> > &matches,
                        vector<vector<int> > &edge1, vector<vector<int> > &edge2,
                        vector<KeyPoint> &kpts1, vector<KeyPoint> &kpts2) {
      Mat img_aux, img_out;
      namedWindow("Hyperedge Matching", WINDOW_NORMAL);
      for (int i = 0; i < matches.size(); i++) {
        hconcat(img1, img2, img_aux);
        cvtColor(img_aux, img_out, CV_GRAY2RGB);
        int base_idx = matches[i].first;
        int ref_idx  = matches[i].second;
        Point2f p0 = kpts1[edge1[base_idx][0]].pt;
        Point2f p1 = kpts1[edge1[base_idx][1]].pt;
        Point2f p2 = kpts1[edge1[base_idx][2]].pt;

        Point2f q0 = kpts2[edge2[ref_idx][0]].pt;
        Point2f q1 = kpts2[edge2[ref_idx][1]].pt;
        Point2f q2 = kpts2[edge2[ref_idx][2]].pt;

        circle(img_out, p0, 2, Scalar(0, 0, 255), 3);
        circle(img_out, p1, 2, Scalar(0, 0, 255), 3);
        circle(img_out, p2, 2, Scalar(0, 0, 255), 3);
        circle(img_out, q0 + Point2f(img1.cols, 0), 2, Scalar(0, 255, 0), 3);
        circle(img_out, q1 + Point2f(img1.cols, 0), 2, Scalar(0, 255, 0), 3);
        circle(img_out, q2 + Point2f(img1.cols, 0), 2, Scalar(0, 255, 0), 3);
        imshow("Hyperedge Matching", img_out);
        waitKey(0);
      }
    }

    void pointsMatch(Mat &img1, vector<KeyPoint> &kpts1, Mat &img2,
                     vector<KeyPoint> &kpts2, vector<DMatch> &matches) {
        Mat out_img;
        namedWindow("Matches", WINDOW_NORMAL);
        drawMatches(img1, kpts1, img2, kpts2, matches, out_img);
        resizeWindow("Matches", 800, 900);
        imshow("Matches", out_img);
        waitKey(0);
    }
}
