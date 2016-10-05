#include "iostream"
#include "stdio.h"
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;


//Distancia ecluidinana entre los puntos de una misma imagen usando descriptores
// double euclidDistance(Mat &vect1){
// Mat matrixEuclidian(vect1.rows, vect1.rows, DataType<double>::type);
// for (int i = 0; i <vect1.rows; i++){
// 	for (int j = 0; j <vect1.rows; j++){
// 		double sum = 0.0; 
// 		for (int k = 0; k < vect1.cols; k++){
// 			sum +=((vect1.at<double>(k, i) - vect1.at<double>(k, j))*(vect1.at<double>(k, i) - vect1.at<double>(k, j)));
// 		}
// 		matrixEuclidian.at<double>(i,j) = sqrt(sum);
// 	}
// }
//  // for (int i = 0; i < vect1.rows; i++) {
//  //    for (int j = 0; j < vect1.rows; j++) {
    	
//  //    		 cout << matrixEuclidian.at<double>(j, i) << " ";
//  //    }
//  //    cout << endl;
//   //}
// }



// //Distancias entre los puntos caracteristicas de una imagen1 usando la posición del punto.
float distancePoints(vector<KeyPoint> &point){
	float n = point.size();
	Mat matrixEuclidian(n, n, DataType<float>::type), matrixEuclidianSort(n, n, DataType<int>::type);
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			float x1 = point[i].pt.x;
			float x2 = point[j].pt.x;
			float y1 = point[i].pt.y;
			float y2 = point[j].pt.y;
			matrixEuclidian.at<float>(i,j) = sqrt(((x1 - x2)*(x1 - x2)) + ((y1 - y2)*(y1 - y2))); 
		}

	}
	//cv::sortIdx(matrixEuclidian, matrixEuclidianSort , CV_SORT_ASCENDING);

	for (int i = 0; i < 10; i++) {
    	for (int j = 0; j < 10; j++) {
    	
    		 cout << matrixEuclidian.at<float>(i,j) << " ";
	    }
	    cout << endl;
	}
	// cout<<"ordenado" <<endl; 
	// for (int i = 0; i < 10; i++) {
 //    	for (int j = 0; j < 10; j++) {
    	
 //    		 cout << matrixEuclidianSort.at<int>(i,j) << " ";
	//     }
	//     cout << endl;
	// }
}






 // distancia del arcoseno. 
 float distanceBetweenImg(Mat &vec1, Mat &vec2){
  Mat angulo(vec1.rows, vec2.rows, DataType<float>::type);
  for (int i = 0; i < vec1.rows; i++) {
    for (int j = 0; j < vec2.rows; j++) {
     float producto = 0.0;
     float norma1 = 0.0;
     float norma2 = 0.0;  
      for (int k = 0; k < vec1.cols; k++) {
      	norma1 += vec1.at<float>(k, i)*vec1.at<float>(k, i) ;
      	norma2 += vec2.at<float>(k, j)*vec2.at<float>(k, j);
        producto   += (vec1.at<float>(k, i)*vec2.at<float>(k, j));
      }

      angulo.at<float>(i, j) = (producto/((sqrt(norma1)*sqrt(norma2))));
    }
  }

 for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
    	//if (angulo.at<double>(i,j)> 0.5000000)
    	//{
    		 cout << angulo.at<float>(j, i) << " ";
    	//}
    }
    cout << endl;
 }
}

/*Algoritmo de los k vecinos más cercanos, esto nos permitira conocer 
los indices de la matrix de los vecinos más cercanos, 
para hacer el hipergrafo de la img1 como de img2 */
float KNN(Mat &matEucl){
	float minDist = 1e6; 
	int minIdx1 = -1; 
	int minIdx2 = -1; 
	for (int i = 0; i < matEucl.rows; i++){
		for (int j = 0; j < matEucl.cols; j++){
			if((matEucl.at<float>(i,j) < minDist)&&(j!=i)){

				minDist = matEucl.at<float>(i,j);
				minIdx1 = j; 
			}

		}
		minDist = 1e6; 
		for (int j = 0; j < matEucl.cols; j++){
			if((matEucl.at<float>(i,j) < minDist) &&(j!=minIdx1) &&(j!=i)){

				minDist = matEucl.at<float>(i,j);
				minIdx2 = j; 
			}	
			//como guardar los indices en una Matriz de N por 3; 
	 }

	}
}


int main(int argc, const char *argv[]){
  const Mat imgA = imread("./house/house.seq0.png", 0);  // Load as grayscale
  const Mat imgB = imread("./house/house.seq0.png", 0); // Load as grayscale

  SiftFeatureDetector detector;

  vector<KeyPoint> keypoints;
  vector<KeyPoint> keypoints1;
  detector.detect(imgA, keypoints);
  detector.detect(imgB, keypoints1);
  Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create("SIFT");

  Mat descriptorA, descriptorB;

  descriptor->compute(imgA, keypoints, descriptorA);
  descriptor->compute(imgB, keypoints1, descriptorB);

  // Add results to image and save.
  Mat output1;
  Mat output;

  drawKeypoints(imgA, keypoints, output);
  imwrite("sift_result.jpg", output);
  drawKeypoints(imgB, keypoints1, output1);
  imwrite("sift_result1.jpg", output);
  distancePoints(keypoints1);
  distanceBetweenImg(descriptorA, descriptorB);
  //euclidDistance(descriptorA);
  //euclidDistance(descriptorB);
  //KNN(euclidDistance(descriptorA)); 

   cout << keypoints.size() << endl;
   cout << keypoints1.size() << endl;

  // for (int i = 0; i < keypoints.size(); i++) {
  //   cout << "DescriptorA (" << descriptorA.row(i) << ")" << endl;
  // }
 

  return 0;
}
