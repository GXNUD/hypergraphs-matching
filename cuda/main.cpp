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
#include <getopt.h>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include "match.hpp"
#include "d_match2.hpp"
#include "draw.hpp"
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace cv;
using namespace std;
using namespace cv::gpu;

typedef struct HyperEdgeMatches{
    int p_idx;
    int q_idx;
    float total_sim;
    float angles_sim,ratios_sim,desc_sim;
}HEM;



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

int descriptorToArray(Mat &descriptor, float *array){
    int rows = descriptor.rows;
    int cols = descriptor.cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i*cols+j] = descriptor.at<float>(i,j);
        }
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


pair<vector<HyperEdgeMatches>, vector<DMatch>> build_hyperedge_matches(beforeMatches
        *beMatches, float th_e,int edges1Size, float th_points, float *desc1,float *desc2){
    vector<HyperEdgeMatches> hyperedge_matches;
    vector<DMatch> point_matches;
    set<pair<int,int>> selected_point_matches;
    for (int i = 0; i < edges1Size; i++) {
        float descsub = 0.0;
        if(beMatches[i].max_similarity>=th_e){
            HyperEdgeMatches cur_match;
            cur_match.p_idx = i;
            cur_match.q_idx = beMatches[i].bestIndex_j;
            cur_match.total_sim = beMatches[i].max_similarity;
            cur_match.angles_sim = beMatches[i].s_ang;
            cur_match.ratios_sim = beMatches[i].s_rat;
            cur_match.desc_sim = beMatches[i].s_desc;
            hyperedge_matches.push_back(cur_match);
            for (int l = 0; l < 3 ; l++) {
                int idx1_m = beMatches[i].edge_match_indices[l].x;
                int idx2_m = beMatches[i].edge_match_indices[l].y;
                for (int ll = 0; ll < 64; ll++) {
                    descsub = descsub + powf((desc1[idx1_m*64+ll] - desc2[idx2_m*64+ll]),2);
                }
                float dist = sqrtf(descsub);
                float points_sim = expf(-dist/0.5);
                if(selected_point_matches.count(make_pair(idx1_m,idx2_m)) == 0 &&
                        points_sim >= th_points){
                    point_matches.push_back(DMatch(idx1_m,idx2_m,dist));
                    selected_point_matches.insert(make_pair(idx1_m,idx2_m));
                }
            }
        }
    }
    pair<vector<HyperEdgeMatches>, vector<DMatch>> edge_and_points_matches;
    edge_and_points_matches = make_pair(hyperedge_matches, point_matches);
    return edge_and_points_matches;
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

int doMatch(Mat &img1, Mat &img2, float cang,
        float crat, float cdesc) {

  clock_t begin = clock();
  GpuMat d_img1(img1);
  GpuMat d_img2(img2);


  //Mat img1 = imread(nameImg1,0);
  //Mat img2 = imread(nameImg2,0);

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

  size_t valor;
  valor = 256*1024*1024;
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, valor);
  valor = 80 * 1024;
  cudaDeviceSetLimit(cudaLimitStackSize, valor);

  cudaDeviceGetLimit(&valor, cudaLimitStackSize);
  cout << "cudaLimitStackSize = " << valor/1024 << "KB" << endl;
  cudaDeviceGetLimit(&valor, cudaLimitMallocHeapSize);
  cout << "cudaLimitMallocHeapSize = " << valor/1024/1024 << "MB" << endl;

  // Conversion to c array
  int *edges1Array = (int*)malloc(3*Edges1.size()*sizeof(int));
  int *edges2Array = (int*)malloc(3*Edges2.size()*sizeof(int));
  vectorVectorToArray(Edges1, edges1Array);
  vectorVectorToArray(Edges2, edges2Array);
  int *d_edges1Array, *d_edges2Array;
  gpuErrchk(cudaMalloc((void**)&d_edges1Array, 3*Edges1.size()*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_edges2Array, 3*Edges2.size()*sizeof(int)));
  gpuErrchk(cudaMemcpy(d_edges1Array, edges1Array, 3*Edges1.size()*sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_edges2Array, edges2Array, 3*Edges2.size()*sizeof(int), cudaMemcpyHostToDevice));


  cout << "tamano kpts1 : " <<kpts1.size()<<endl;
  cout << "tamano descriptors : "<< descriptor1.size() << " "<< descriptor2.size() << endl;

  // Conversion of KeyPoint to array
  // La columna cero del arreglo hace referencia a X y la 1 a Y
  float *keyPoints1Array, *keyPoints2Array;
  keyPoints1Array = (float*)malloc(kpts1.size()*sizeof(float)*2);
  keyPoints2Array = (float*)malloc(kpts2.size()*sizeof(float)*2);
  keyPointsToArray(kpts1, keyPoints1Array);
  keyPointsToArray(kpts2, keyPoints2Array);
  float *d_keyPoints1Array, *d_keyPoints2Array;
  gpuErrchk(cudaMalloc((void**)&d_keyPoints1Array, kpts1.size()*sizeof(float)*2));
  gpuErrchk(cudaMalloc((void**)&d_keyPoints2Array, kpts2.size()*sizeof(float)*2));
  gpuErrchk(cudaMemcpy(d_keyPoints1Array, keyPoints1Array, kpts1.size()*sizeof(float)*2, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_keyPoints2Array, keyPoints2Array, kpts2.size()*sizeof(float)*2, cudaMemcpyHostToDevice));


  // Conversion of descriptors to array
  float *descriptor1Array, *descriptor2Array;
  descriptor1Array = (float*)malloc(descriptor1.rows*descriptor1.cols*sizeof(float));
  descriptor2Array = (float*)malloc(descriptor2.rows*descriptor2.cols*sizeof(float));
  descriptorToArray(descriptor1, descriptor1Array);
  descriptorToArray(descriptor2, descriptor2Array);
  float *d_descriptor1Array, *d_descriptor2Array;
  gpuErrchk(cudaMalloc((void**)&d_descriptor1Array, descriptor1.rows*descriptor1.cols*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_descriptor2Array, descriptor2.rows*descriptor2.cols*sizeof(float)));
  gpuErrchk(cudaMemcpy(d_descriptor1Array, descriptor1Array, descriptor1.rows*descriptor1.cols*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_descriptor2Array, descriptor2Array, descriptor2.rows*descriptor2.cols*sizeof(float), cudaMemcpyHostToDevice));

//  int *gpu_matches, *d_matches;
//  gpu_matches = (int*)malloc(100*sizeof(int)*2);
  //gpuErrchk(cudaMalloc((void**)&d_matches,100*sizeof(int)*2));

  float *d_tests, *tests;
  tests = (float*)malloc(Edges1.size()*Edges2.size()*sizeof(float));
  gpuErrchk(cudaMalloc((void**)&d_tests,sizeof(float)*Edges1.size()*Edges2.size()));

  beforeMatches *d_beMatches, *beMatches;
  beMatches = (beforeMatches*)malloc(Edges1.size()
          *sizeof(beforeMatches));
  gpuErrchk(cudaMalloc((void**)&d_beMatches,Edges1.size()
              *sizeof(beforeMatches)));


  float sizeX = (float)Edges1.size();
  //float sizeY = (float)Edges1.size();
  dim3 dimGrid(ceil(sizeX/16.0),1/*ceil(sizeY/16.0)*/,1);
  dim3 dimBlock(16,1,1);
  d_hyperedges<<<dimGrid,dimBlock>>> (d_edges1Array, d_edges2Array, d_keyPoints1Array, d_keyPoints2Array,
        d_descriptor1Array, d_descriptor2Array, descriptor1.rows, descriptor1.cols,
        descriptor2.rows, descriptor2.cols, cang, crat, cdesc, 0.75,
        Edges1.size(), Edges2.size(), d_beMatches, d_tests);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(beMatches, d_beMatches, Edges1.size()
              *sizeof(beforeMatches), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(tests, d_tests, Edges1.size()*Edges2.size()
              *sizeof(float), cudaMemcpyDeviceToHost));
  cout << "sin test "<< beMatches[0].edge_match_indices[2].y<<endl;

  FILE *fileTest,*fileTest2;
  fileTest = fopen("sim_anglesTest","w");
  fprintf(fileTest,"bestIndex_j,punto1_x,punto1_y,punto2_x,punto2_y,punto3_x,punto3_y,maxSimilarity,s_ang,s_rat,s_desc\n");
  for (int i = 0; i < Edges1.size(); i++) {
      fprintf(fileTest,"%d,%d,%d,%d,%d,%d,%d,%0.2f,%0.2f,%0.2f,%0.2f\n", beMatches[i].bestIndex_j, beMatches[i].edge_match_indices[0].x,
              beMatches[i].edge_match_indices[0].y,  beMatches[i].edge_match_indices[1].x,
              beMatches[i].edge_match_indices[1].y, beMatches[i].edge_match_indices[2].x, beMatches[i].edge_match_indices[2].y,
              beMatches[i].max_similarity,beMatches[i].s_ang,beMatches[i].s_rat,beMatches[i].s_desc);
  }
  fclose(fileTest);

  fileTest2 = fopen("tests","w");
  for (int i = 0; i < Edges1.size(); i++) {
      for (int j = 0; j < Edges2.size(); j++) {
          fprintf(fileTest2,"%f ",tests[i*Edges2.size()+j]);
      }
      fprintf(fileTest2,"\n");
  }
  fclose(fileTest2);

  HyperEdgeMatches *hyperedge_matches;
  hyperedge_matches = (HyperEdgeMatches*)malloc(Edges1.size()
          *sizeof(HyperEdgeMatches));
  pair<vector<HyperEdgeMatches>, vector<DMatch>> edge_and_points_matches;
  edge_and_points_matches = build_hyperedge_matches(beMatches,0.7,Edges1.size(),0.7,descriptor1Array, descriptor2Array);

  vector<HyperEdgeMatches> edgetestmatches = edge_and_points_matches.first;
  vector<DMatch> point_matches = edge_and_points_matches.second;

  cout << "Número de Matches " << point_matches.size() << endl;


  cout << Edges1.size() << " Edges from image 1" << endl;
  cout << Edges2.size() << " Edges from image 2" << endl;

  // Draw Point matching
  draw::pointsMatch(img1, kpts1, img2, kpts2, point_matches);

  free(edges1Array); free(keyPoints1Array); free(keyPoints2Array);
  free(descriptor1Array); free(descriptor2Array);
  cudaFree(d_edges1Array); cudaFree(d_edges2Array);
  cudaFree(d_keyPoints1Array); cudaFree(d_keyPoints2Array);
  cudaFree(d_descriptor1Array); cudaFree(d_descriptor2Array);
  free(beMatches); cudaFree(d_beMatches);
  free(hyperedge_matches);free(tests);cudaFree(d_tests);
  return 0;
}

void cright() {
  cout << "Sample implementation of LYSH algorithm for image matching" << endl;
  cout << "Copyright (C) 2016 L.A. Campeon, Y.H. Gomez, J.S. Vega, J.H. Osorio." << endl;
  cout << "This is free software; see the source code for copying conditions." << endl;
  cout << "There is ABSOLUTELY NO WARRANTY; not even for MERCHANTABILITY or" << endl;
  cout << "FITNESS FOR A PARTICULAR PURPOSE." << endl;
  cout << endl;
}


void usage(char* program_name) {
  int n = 3;
  string opts[] = {"--cang", "--crat", "--cdesc"};
  string description[] = {
    "Constant of angle similarity (default: 1)",
    "Constant of ratio similarity (default: 1)",
    "Constant of SURF descriptor similarity (default: 1)"
  };
cout << "Usage: " << program_name << " [options ...] img1 img2" << endl;
  cout << endl;
  cout << "Matching options" << endl;
  for (int i = 0; i < n; i++) {
    cout << "  " << opts[i] << ": " << description[i] << endl;
  }

  exit(EXIT_FAILURE);
}

pair<bool, float> toFloat(string s) {
  stringstream ss(s);
  float x;
  ss >> x;
  if (!ss) {
    return make_pair(false, 0);
  }
  return make_pair(true, x);
}

int main(int argc, char *argv[]) {
  int opt, opt_index = 0;
  static struct option options[] = {
    {"cang", required_argument, 0, 'a'},
    {"crat", required_argument, 0, 'r'},
    {"cdesc", required_argument, 0, 'd'},
    {0, 0, 0, 0}
  };

  double cang = 1, crat = 1, cdesc = 1;
  pair<bool, double> convert_type(true, 0);
  while ((opt = getopt_long(argc, argv, "a:r:d:", options, &opt_index)) != -1) {
    switch (opt) {
      case 'a':
        convert_type = toFloat(optarg);
        cang = convert_type.second;
        break;
      case 'r':
        convert_type = toFloat(optarg);
        crat = convert_type.second;
        break;
      case 'd':
        convert_type = toFloat(optarg);
        cdesc = convert_type.second;
        break;
      default:
        usage(argv[0]);
        break;
    }
  }

  if (!convert_type.first) {
    usage(argv[0]);
  }

  if (argc - optind != 2) {
    cout << "Error: You must provide two images" << endl << endl;
    usage(argv[0]);
  }

  vector<Mat> img(2);
  for (int i = optind, j = 0; i < argc; i++, j++) {
    img[j] = imread(argv[i], CV_LOAD_IMAGE_GRAYSCALE);
    if (!img[j].data) {
      cout << "Error: img1 and img2 must be valid images both" << endl << endl;
      usage(argv[0]);
    }
  }

  doMatch(img[0], img[1], cang, crat, cdesc);

  return 0;
}
