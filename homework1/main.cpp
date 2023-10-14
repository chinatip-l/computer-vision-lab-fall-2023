#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <string.h>
#include "include/func.hpp"

#define MAX_LEN 250
#define BASE_PATH "/home/chinatip/work/computer_vision/homework1"
#define SRC_PATH "test_img"
#define OUT_PATH "output"
#define FILENAME "taipei101"
// #define FILENAME "aeroplane"
// #define FILENAME "test"
#define BW_OUT_SUFX "Q1"
#define EDGE_OUT_SUFX "Q2"
#define POOL_OUT_SUFX "Q3"
#define BIN_OUT_SUFX "Q4"
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

    Mat bw_img;
    bw_img = applyGreyscaleFilter(og_img);
    printf("BW Img Size: w x h %d x %d\n", bw_img.cols, bw_img.rows);

    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s.%s", BASE_PATH, OUT_PATH, FILENAME, BW_OUT_SUFX, EXT);
    imwrite(out_file, bw_img);

    int raw_kernel[] = {-1, -1, -1,
                        -1, 8, -1,
                        -1, -1, -1};

    Mat kernel(3, 3, CV_32S, raw_kernel);
    Mat edge;
    edge = applyConvolve(bw_img, kernel);
    printf("Edge Img Size: w x h %d x %d\n", edge.cols, edge.rows);

    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s.%s", BASE_PATH, OUT_PATH, FILENAME, EDGE_OUT_SUFX, EXT);
    imwrite(out_file, edge);

    Mat pooled;
    pooled = applyMaxPooling(edge, 2,2, 2);
    printf("Pooled Img Size: w x h %d x %d\n", pooled.cols, pooled.rows);

    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s.%s", BASE_PATH, OUT_PATH, FILENAME, POOL_OUT_SUFX, EXT);
    imwrite(out_file, pooled);

    Mat bin;
    int thres=64;
    bin = applyBinarisation(pooled, thres);
    printf("Binarised (T=%d) Img Size: w x h %d x %d\n",thres, bin.cols, bin.rows);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_t_%d.%s", BASE_PATH, OUT_PATH, FILENAME, BIN_OUT_SUFX,thres, EXT);
    imwrite(out_file, pooled);

    namedWindow("Original", WINDOW_AUTOSIZE);
    appendImgToCanvas(og_img);
    appendImgToCanvas(bw_img);
    appendImgToCanvas(edge);
    appendImgToCanvas(pooled);
    appendImgToCanvas(bin);
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
        Size s(canvas.cols + img.cols + 5, canvas.rows);
        size_t old_w = canvas.cols;
        copyMakeBorder(canvas, canvas, 0, 0, 0, img.cols + 5, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
        img.copyTo(canvas(Rect(old_w + 5, 0, img.cols, img.rows)));
    }
}

