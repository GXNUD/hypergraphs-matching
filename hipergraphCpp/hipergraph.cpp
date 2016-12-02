#include "iostream"
#include "stdio.h"
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

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
  const Mat imgA = imread("./house/house.seq0.png", 0); // Load as grayscale
  const Mat imgB = imread("./house/house.seq0.png", 0); // Load as grayscale

  // Mat prueba2(4, 3, DataType<float>::type);

  // prueba 2
  // prueba2.at<int>(0, 0) = 0;
  // prueba2.at<int>(0, 1) = 1;
  // prueba2.at<int>(0, 2) = 2;
  //
  // prueba2.at<int>(1, 0) = 2;
  // prueba2.at<int>(1, 1) = 0;
  // prueba2.at<int>(1, 2) = 1;
  //
  // prueba2.at<int>(2, 0) = 2;
  // prueba2.at<int>(2, 1) = 1;
  // prueba2.at<int>(2, 2) = 0;
  //
  // prueba2.at<int>(3, 0) = 1;
  // prueba2.at<int>(3, 1) = 4;
  // prueba2.at<int>(3, 2) = 3;

  SiftFeatureDetector detector(4);

  vector<KeyPoint> keypoints1;
  vector<KeyPoint> keypoints2;
  detector.detect(imgA, keypoints1);
  detector.detect(imgB, keypoints2);
  Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create("SIFT");

  Mat descriptorA, descriptorB;

  descriptor->compute(imgA, keypoints1, descriptorA);
  descriptor->compute(imgB, keypoints2, descriptorB);
  // cout << "Descriptor size " << descriptorA.cols << endl;
  // for (int i = 0; i < descriptorA.cols; i++) {
  //   cout << "Index " << i << ": "<< descriptorA.at<double>(0, i) << endl;
  // }
  // cout << endl;
  // Add results to image and save.
  Mat output1;
  Mat output2;

  drawKeypoints(imgA, keypoints1, output1);
  imwrite("sift_result.jpg", output1);
  drawKeypoints(imgB, keypoints2, output2);
  imwrite("sift_result1.jpg", output2);
  Mat dist1 = distancePoints(keypoints1);
  // for (int i = 0; i < dist1.rows; i++) {
  //   for (int j = 0; j < dist1.cols; j++) {
  //     cout << "Dist " << i << " and " << j << ": " << dist1.at<double>(i, j) << endl;
  //   }
  // }
  Mat dist2 = distancePoints(keypoints2);
  // cout << endl << "Distance second image" << endl;
  // for (int i = 0; i < dist2.rows; i++) {
  //   for (int j = 0; j < dist2.cols; j++) {
  //     cout << "Dist " << i << " and " << j << ": " << dist2.at<double>(i, j) << endl;
  //   }
  // }
  Mat matSimilarity = distanceBetweenImg(descriptorA, descriptorB);
  for (int i = 0; i < matSimilarity.rows; i++) {
    for (int j = 0; j < matSimilarity.cols; j++) {
      cout << "Similarity " << i << " and " << j << ": " << matSimilarity.at<double>(i, j) << endl;
    }
  }
  // Mat dist1 = euclidDistance(descriptorA);
  // for (int i = 0; i < dist1.rows; i++) {
  //   for (int j = 0; j < dist1.cols; j++) {
  //     cout << dist1.at<double>(i, j) << " ";
  //   }
  //   cout << endl;
  // }
  // euclidDistance(descriptorB);
  // KNN(prueba);
  // Mat prueba2(keypoints.size(), 3, DataType<float>::type);
  // Mat Edges1 = KNN(dist1);
  // Mat Edges2 = KNN(dist2);
  // positionXYIJK(prueba2,keypoints);
  //
  // cout << keypoints.size() << endl;
  // cout << keypoints1.size() << endl;

  // for (int i = 0; i < keypoints.size(); i++) {
  //   cout << "DescriptorA (" << descriptorA.row(i) << ")" << endl;
  // }

  return 0;
}
