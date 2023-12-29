#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <string.h>
#include "include/func.hpp"

#define MAX_LEN 250

// path for lab
#define BASE_PATH "/home/chinatip/computer-vision-lab-fall-2023/homework4"
#define SRC_PATH "test_img"
#define OUT_PATH "output"

// define file name, can uncomment to select the input
#define FILENAME "img3"
// #define FILENAME "img2"
// #define FILENAME "img3"
#define GAUSSIAN_FILT_SUFX "q1"
#define CANNY_EDGE_SUFX "q2"
#define HOUGH_SUFX "q3"
#define KER_INFO_3 "K3P1"
#define KER_INFO_5 "K5P2"
#define KER_INFO_7 "K7P3"
#define EXT "jpg"



Mat canvas;

void appendImgToCanvas(Mat);
void drawArrows(cv::Mat &, const cv::Mat &, const cv::Mat &, int , double);

int main(int argc, char **argv)
{

    Mat og_img;
    char og_file[MAX_LEN] = "", out_file[MAX_LEN] = "";
    snprintf(og_file, MAX_LEN, "%s/%s/%s.%s", BASE_PATH, SRC_PATH, FILENAME, EXT);

    og_img = imread(og_file);

    if (!og_img.data)
    {
        printf("No image data \n");
        return -1;
    }
    else
    {
        printf("Original Img Size: w x h %d x %d\n", og_img.cols, og_img.rows);
    }

    appendImgToCanvas(og_img);
    Mat bw_img;
    bw_img = applyGreyscaleFilter(og_img);

    Mat res_img, selected_img;
    
    Mat gaussian_filter = createGaussianFilter(13, 5);
    res_img = applyFilter(bw_img, gaussian_filter, 1, false);
    printf("Gaussian Filter Img Size: w x h %d x %d\n", res_img.cols, res_img.rows);
    
    appendImgToCanvas(res_img);

    Mat sobelx;
    sobelx = applySobelX(res_img);
    appendImgToCanvas(sobelx);
    

    Mat sobely;
    sobely = applySobelY(res_img);
    appendImgToCanvas(sobely);
    

    Mat magnitude;
    magnitude = calculateEdgeStrength(sobelx, sobely);
    // GaussianBlur(magnitude, magnitude,Size(5,5),1,1);
    // appendImgToCanvas(magnitude);

    Mat direction;
    direction=calculateEdgeDirection(sobelx,sobely);
    imshow("Original",direction);
    // waitKey(2000);

    vector<Point2f> contour;

    contour=initContourPointCircle(magnitude.cols/2,magnitude.rows/2,300,100);
    float alpha=0.5,beta=0.01,gamma=0.1;

    vector<Mat> buf;
    for(int k =0; k<5000;k++){
        Mat i;
        activeContour(magnitude,direction,contour,alpha,beta,gamma);
        i=showSnake(magnitude,contour);
        // buf.push_back(i);
        // appendImgToCanvas(i);
        imshow("Original", i);
        waitKey(10);
        // printf("Frame #%d\n",k);
    }
    printf("Finish\n");
    // appendImgToCanvas(magnitude);
    // appendImgToCanvas(i);


    
    namedWindow("Original", WINDOW_AUTOSIZE);
    // imshow("Original", canvas);
    waitKey(0);
    return 0;
}

void appendImgToCanvas(Mat img)
{
    if (canvas.empty())
    {
        canvas = img;
    }
    else
    {
        Size s(canvas.cols + img.cols + 5, canvas.rows > img.rows ? canvas.rows : img.rows);
        size_t old_w = canvas.cols;
        copyMakeBorder(canvas, canvas, 0, 0, 0, img.cols + 5, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
        img.copyTo(canvas(Rect(old_w + 5, 0, img.cols, img.rows)));
    }
}


