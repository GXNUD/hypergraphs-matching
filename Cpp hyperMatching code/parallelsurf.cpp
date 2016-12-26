#include "iostream"
#include "stdio.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/nonfree/gpu.hpp"
#include <opencv2/nonfree/features2d.hpp>
using namespace cv;
using namespace std;
using namespace gpu;


int main(int argc, const char *argv[]){

  //lectura de las imagenes sobre la gpu.
  GpuMat img1(imread("./house/house.seq0.png", 0));
  if (img1.data == NULL) {
		printf("cv::imread() failed...\n");
		return -1;
  }
  GpuMat img2(imread("./house/house.seq0.png", 0));
  if (img2.data == NULL) {
		printf("cv::imread() failed...\n");
		return -1;
  }
  //Estracción de caracteristicas con surf
  SURF_GPU surf(400);
  GpuMat keypoints1GPU, keypoints2GPU;
  GpuMat descriptors1GPU, descriptors2GPU;
  surf(img1, gpu::GpuMat(), keypoints1GPU, descriptors1GPU);
  surf(img2, gpu::GpuMat(), keypoints2GPU, descriptors2GPU);
  // estraccion de puntos caracteristicas con la gpu
  cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
  cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;
  // paso de la información de la gpu a la CPU
  vector< KeyPoint> keypoints1, keypoints2;
  vector< float> descriptors1, descriptors2;
  surf.downloadKeypoints(keypoints1GPU, keypoints1);
  surf.downloadKeypoints(keypoints2GPU, keypoints2);
  surf.downloadDescriptors(descriptors1GPU, descriptors1);
  surf.downloadDescriptors(descriptors2GPU, descriptors2);
  
  // Dibujo de las imagenes de los puntos caracteriticos.
  Mat output1;
  Mat output2;
  Mat img11, img22;
  img1.download(img11);
  img2.download(img22);

  drawKeypoints(img11, keypoints1, output1);
  imwrite("surfgpu_result1.jpg", output1);
  drawKeypoints(img22, keypoints2, output2);
  imwrite("surftgpu_result2.jpg", output2);



  return 0;
}
