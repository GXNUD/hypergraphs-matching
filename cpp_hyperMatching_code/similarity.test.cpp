#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "similarity.hpp"
using namespace std;
using namespace cv;

Point2f rot(Point2f p, double t) {
    double x = p.x;
    double y = p.y;
    return Point2f(x * cos(t) - y * sin(t), x * sin(t) + y * cos(t));
}

Point2f trans(Point2f p, double tx, double ty) {
    double x = p.x;
    double y = p.y;
    return Point2f(x + tx, y + ty);
}

int main(int argc, char* argv[]) {
    vector<Point2f> p, q;
    p.push_back(Point2f(0, 0));
    p.push_back(Point2f(1, 1));
    p.push_back(Point2f(-1, 1));
    // for (int i = 0; i < 3; i++) {
    //     q.push_back(trans(rot(p[i], 10 * M_PI / 180), 2, 3));
    // }

    q.push_back(Point2f(1, 0));
    q.push_back(Point2f(1, -1));
    q.push_back(Point2f(3, 2));

    double data1[][3] = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}
    };
    double data2[][3] = {
        {4, 8, 3}, {7, 5, 0}, {1, 4, 6}
    };
    vector<Mat> d1, d2;
    for (int i = 0; i < 3; i++) {
        d1.push_back(Mat(1, 3, CV_64FC1, &data1[i]));
        d2.push_back(Mat(1, 3, CV_64FC1, &data2[i]));
    }


    for (int i = 0; i < 3; i++) {
        cout << "p" << i << ": " << p[i] << endl;
        cout << "    " << d1[i] << endl;
    }
    for (int i = 0; i < 3; i++) {
        cout << "q" << i << ": " << q[i] << endl;
        cout << "    " << d2[i] << endl;
    }

    cout << sim::angles(p, q) << endl;
    cout << sim::ratios(p, q) << endl;
    cout << sim::descriptors(d1, d2) << endl;

    return 0;
}
