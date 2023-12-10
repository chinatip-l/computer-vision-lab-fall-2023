#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <string.h>
#include "include/func.hpp"

#define MAX_LEN 250
// // path for personal computer
// #define BASE_PATH "/home/chinatip/work/computer_vision/homework3"

// path for lab
#define BASE_PATH "/home/chinatip/computer-vision-lab-fall-2023/homework3"
#define SRC_PATH "test_img"
#define OUT_PATH "output"

// define file name, can uncomment to select the input
#define FILENAME "img1"
// #define FILENAME "img2"
// #define FILENAME "img3"
#define GAUSSIAN_FILT_SUFX "q1"
#define CANNY_EDGE_SUFX "q2"
#define HOUGH_SUFX "q3"
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
    Mat bw_img;
    // bw_img=applyGreyscaleFilter(og_img);
    bw_img=og_img;

    Mat res_img,selected_img;
    int kernel_size;
    float sigma;
    for(sigma=0.5;sigma<=1.0;sigma+=0.5){
        for(kernel_size=5;kernel_size<=5;kernel_size+=2){
            Mat gaussian_filter=createGaussianFilter(kernel_size,sigma);
            res_img = applyFilter(bw_img,  gaussian_filter, 1,false);
            printf("Gaussian Filter Img Size: w x h %d x %d\n", res_img.cols, res_img.rows);
            snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_K%d_SIG_%.1f.%s", BASE_PATH, OUT_PATH, FILENAME, GAUSSIAN_FILT_SUFX,kernel_size,sigma, EXT);
            imwrite(out_file, res_img);
            // appendImgToCanvas(res_img);
            if(sigma==1.0 && kernel_size == 5){
                selected_img=res_img.clone();
                
            }
        }
    }
    appendImgToCanvas(selected_img);

    Mat sobelx;
    sobelx=applySobelX(selected_img);
    // sobelx = sobelx.mul(sobelx);

    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_SOBEL_X.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, sobelx);
    appendImgToCanvas(sobelx);
    Mat sobely;
    sobely=applySobelY(selected_img);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_SOBEL_Y.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, sobely);
    appendImgToCanvas(sobely);

    Mat edge_strength;
    edge_strength=calculateEdgeStrength(sobelx,sobely);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_EDGE_STRENGTH.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, edge_strength);
    appendImgToCanvas(edge_strength);
    Mat edge_direction;
    edge_direction=calculateEdgeDirection(sobelx,sobely);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_EDGE_DIR.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, edge_direction);
    // appendImgToCanvas(edge_direction);

    Mat nms;
    nms=calculateNonMaximumSuppression(edge_strength,edge_direction);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_NMS.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, nms);
    appendImgToCanvas(nms);



    


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

