#ifndef _FUNC_H_
#define _FUNC_H_

#include <opencv2/opencv.hpp>

using namespace cv;

Mat applyGreyscaleFilter(Mat);
Mat applyConvolve(Mat, Mat);
Mat applyConvolve(Mat, Mat, int, int);
Mat applyMaxPooling(Mat, int, int, int);
Mat applyBinarisation(Mat, int);

#endif