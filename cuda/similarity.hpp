#include <vector>
#include <cmath>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "combination.hpp"

using namespace cv;
using namespace std;

//int testing = 0;

float vectorsAngleSin(Point2f &pivot, Point2f &p, Point2f &q) {
    Point2f v1 = p - pivot;
    Point2f v2 = q - pivot;
    float dot = v1.dot(v2);
    //if (testing == 0 )
      //  cout << " Sin real" << dot << endl;
    //testing++;

    float angle = acos(dot / sqrt((v1.x*v1.x)+(v1.y*v1.y)) / sqrt((v2.x*v2.x)+(v2.y*v2.y)));
    return sin(angle);
}

vector<float> getAnglesSin(vector<Point2f> &p) {
    Point2f &p1 = p[0];
    Point2f &p2 = p[1];
    Point2f &p3 = p[2];
    vector<float> sines(3);
    sines[0] = vectorsAngleSin(p1, p2, p3);
    sines[1] = vectorsAngleSin(p2, p1, p3);
    sines[2] = vectorsAngleSin(p3, p1, p2);
    return sines;
}

bool compMat(Mat &a, Mat &b) {
    return norm(a) < norm(b);
}

namespace sim {
    float angles(vector<Point2f> &p, vector<Point2f> &q, float sigma = 0.5) {
        vector<float> sines1 = getAnglesSin(p);
        vector<float> sines2 = getAnglesSin(q);

        //sort(sines1.begin(), sines1.end());
        float min_diff_between_sin = 1E30;
        do {
            float sum = 0;
            for (int i = 0; i < 3; i++)
                sum += fabs(sines1[i] - sines2[i]);
            min_diff_between_sin = min(min_diff_between_sin, sum);
        } while (next_permutation(sines1.begin(), sines1.end()));

        return exp(-min_diff_between_sin / sigma);
    }

    float ratios(vector<Point2f> &p, vector<Point2f> &q, float sigma = 0.5) {
        vector<vector<int> > idx_perm = getCombination(3, 2);
        vector<float> sides_p;
        vector<float> sides_q;
        for (int k = 0; k < idx_perm.size(); k++) {
            int i = idx_perm[k][0];
            int j = idx_perm[k][1];
            float side_p = norm(p[i] - p[j]);
            float side_q = norm(q[i] - q[j]);
            sides_p.push_back(side_p);
            sides_q.push_back(side_q);
        }

        sort(sides_q.begin(), sides_q.end());
        float min_err = 1E30;
        do {
            vector<float> r(3);
            float sum = 0;
            for (int i = 0; i < 3; ++i) {
                r[i] = sides_p[i] / sides_q[i];
            }

            for (int k = 0; k < idx_perm.size(); k++) {
                sum += fabs(r[idx_perm[k][0]] - r[idx_perm[k][1]]);
            }
            min_err = min(sum, min_err);
        } while (next_permutation(sides_q.begin(), sides_q.end()));
        return exp(- min_err / sigma);
    }

    float descriptors(vector<Mat> &desc1, vector<Mat> &desc2, float sigma = 0.5) {
        float min_diff = 1E30;
        sort(desc2.begin(), desc2.end(), compMat);
        do {
            float diff = 0;
            for (int i = 0; i < 3; i++) {
                diff += norm(desc2[i] - desc1[i]);
            }
            min_diff = min(diff, min_diff);
        } while (next_permutation(desc2.begin(), desc2.end(), compMat));
        return exp(- min_diff / sigma);
    }
}
