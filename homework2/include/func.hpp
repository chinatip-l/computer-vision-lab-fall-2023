#ifndef _FUNC_H_
#define _FUNC_H_

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

Mat applyMeanFilter(Mat,int,int,int);
Mat applyMedianFilter(Mat,int,int,int);
static std::vector<int32_t> sort(std::vector<int32_t>);

#endif