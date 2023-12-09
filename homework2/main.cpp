#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <string.h>
#include "include/func.hpp"

#define MAX_LEN 250
#define BASE_PATH "/home/chinatip/work/computer_vision/homework2"
#define SRC_PATH "test_img"
#define OUT_PATH "output"

// define file name, can uncomment to select the input
#define FILENAME "noise1"
// #define FILENAME "noise2"
#define MEAN_FILT_SUFX "q1"
#define MEDIAN_FILT_SUFX "q2"
#define KER_INFO_3 "K3P1"
#define KER_INFO_5 "K5P2"
#define KER_INFO_7 "K7P3"
#define EXT "png"

using namespace cv;

Mat canvas;

void appendImgToCanvas(Mat);

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

    Mat res_img;
    res_img = applyMedianFilter(og_img,3,1,1);
    printf("Median Filter Img Size: w x h %d x %d\n", res_img.cols, res_img.rows);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s.%s", BASE_PATH, OUT_PATH, FILENAME, MEDIAN_FILT_SUFX, EXT);
    imwrite(out_file, res_img);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_%s.%s", BASE_PATH, OUT_PATH, FILENAME, MEDIAN_FILT_SUFX,KER_INFO_3, EXT);
    imwrite(out_file, res_img);
    appendImgToCanvas(res_img);

    res_img = applyMedianFilter(og_img,5,2,1);
    printf("Median Filter Img Size: w x h %d x %d\n", res_img.cols, res_img.rows);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_%s.%s", BASE_PATH, OUT_PATH, FILENAME, MEDIAN_FILT_SUFX,KER_INFO_5, EXT);
    imwrite(out_file, res_img);
    appendImgToCanvas(res_img);

    res_img = applyMedianFilter(og_img,7,3,1);
    printf("Median Filter Img Size: w x h %d x %d\n", res_img.cols, res_img.rows);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_%s.%s", BASE_PATH, OUT_PATH, FILENAME, MEDIAN_FILT_SUFX,KER_INFO_7, EXT);
    imwrite(out_file, res_img);
    appendImgToCanvas(res_img);

    namedWindow("Original", WINDOW_AUTOSIZE);
    
    imshow("Original", canvas);
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
        Size s(canvas.cols + img.cols + 5, canvas.rows>img.rows?canvas.rows:img.rows);
        size_t old_w = canvas.cols;
        copyMakeBorder(canvas, canvas, 0, 0, 0, img.cols + 5, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
        img.copyTo(canvas(Rect(old_w + 5, 0, img.cols, img.rows)));
    }
}

