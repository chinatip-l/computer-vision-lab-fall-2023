#ifndef _FUNC_H_
#define _FUNC_H_

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

Mat applyGreyscaleFilter(Mat);
float calculateGaussian(int,int,float);

Mat createGaussianFilter(int,float);

Mat applyFilter(Mat , Mat , int ,bool);

Mat applySobelX(Mat);
Mat applySobelY(Mat);

Mat calculateEdgeStrength(Mat , Mat );
Mat calculateEdgeDirection(Mat , Mat );

int quantiseDirection(float);
Mat calculateNonMaximumSuppression(Mat , Mat ) ;

#endif