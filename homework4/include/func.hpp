#ifndef _FUNC_H_
#define _FUNC_H_

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class ContourPoint : public Point2f{
    public:
        float energy;
        bool settle;
};

Mat applyGreyscaleFilter(Mat);
float calculateGaussian(int,int,float);

Mat createGaussianFilter(int,float);

Mat applyFilter(Mat , Mat , int ,bool);

Mat applySobelX(Mat);
Mat applySobelY(Mat);

Mat calculateEdgeStrength(Mat , Mat );
Mat calculateEdgeDirection(Mat , Mat );

vector<ContourPoint> initContourPointCircle(int,int,int,int);

void activeContour(Mat,Mat, vector<ContourPoint>&,float,float,float);
void activeContourWithForce(Mat,Mat,Mat,Mat, vector<ContourPoint>&,float,float,float);

Mat showSnake(Mat, vector<ContourPoint>);
void showGradient(Mat&,Mat,Mat,int);


#endif