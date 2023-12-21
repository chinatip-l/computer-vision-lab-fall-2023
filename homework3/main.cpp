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
    bw_img = applyGreyscaleFilter(og_img);

    Mat res_img, selected_img;
    int kernel_size;
    float sigma;
    for (sigma = 0.5; sigma <= 2; sigma += 0.5)
    {
        for (kernel_size = 3; kernel_size <= 7; kernel_size += 2)
        {
            Mat gaussian_filter = createGaussianFilter(kernel_size, sigma);
            res_img = applyFilter(bw_img, gaussian_filter, 1, false);
            printf("Gaussian Filter Img Size: w x h %d x %d\n", res_img.cols, res_img.rows);
            snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_K%d_SIG_%.1f.%s", BASE_PATH, OUT_PATH, FILENAME, GAUSSIAN_FILT_SUFX, kernel_size, sigma, EXT);
            imwrite(out_file, res_img);
            if (sigma == 1.0 && kernel_size == 5)
            {
                selected_img = res_img.clone();
            }
        }
    }
    // appendImgToCanvas(selected_img);

    Mat sobelx;
    sobelx = applySobelX(selected_img);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_SOBEL_X.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, sobelx);
    // appendImgToCanvas(sobelx);

    Mat sobely;
    sobely = applySobelY(selected_img);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_SOBEL_Y.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, sobely);
    // appendImgToCanvas(sobely);

    Mat edge_strength;
    edge_strength = calculateEdgeStrength(sobelx, sobely);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_EDGE_STRENGTH.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, edge_strength);
    // appendImgToCanvas(edge_strength);

    Mat edge_direction;
    edge_direction = calculateEdgeDirection(sobelx, sobely);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_EDGE_DIR.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, edge_direction);
    // appendImgToCanvas(edge_direction);

    Mat nms;
    nms = calculateNonMaximumSuppression(edge_strength, edge_direction);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_NMS.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, nms);
    // appendImgToCanvas(nms);

    Mat double_thresholed;
    double_thresholed = applyDoubleThresholding(nms, 10, 20);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_DTS.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, double_thresholed);
    // appendImgToCanvas(double_thresholed);

    Mat hyst;
    hyst = applyHysteresis(double_thresholed);
    snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_HYS.%s", BASE_PATH, OUT_PATH, FILENAME, CANNY_EDGE_SUFX, EXT);
    imwrite(out_file, hyst);
    appendImgToCanvas(hyst);

    // make a copy of hysteresis to use as result
    Mat hough = hyst.clone();
    std::vector<cv::Vec2f> ulines;

    // Parameters for Hough Transform
    float rho = 1; // distance resolution in pixels
    int theta_steps[] = {1, 3, 5, 10};
    int thresholds[] = { 35, 50, 75};
    for (int step_theta = 0; step_theta < 4; step_theta++)
    {
        for (int step_threshold = 0; step_threshold < 3; step_threshold++)
        {
            float theta = theta_steps[step_theta] * CV_PI / 180;
            int threshold = thresholds[step_threshold];
            // reset image
            hough = hyst.clone();

            // Apply Hough Transform
            applyHoughTransform(hyst, ulines, rho, theta, threshold);

            // Draw the lines on the original image
            for (size_t i = 0; i < ulines.size(); i++)
            {
                float r = ulines[i][0], t = ulines[i][1];
                double cos_t = std::cos(t), sin_t = std::sin(t);
                double x0 = r * cos_t, y0 = r * sin_t;
                double alpha = 2000; // length that the line will be drawn to cover the image

                // Adjust the start and end points for the new origin at the center of the image
                cv::Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));
                cv::Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));
                cv::line(hough, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
            snprintf(out_file, MAX_LEN, "%s/%s/%s_%s_HOUGH_THETA_%d_THRES_%d.%s", BASE_PATH, OUT_PATH, FILENAME, HOUGH_SUFX,theta_steps[step_theta],threshold, EXT);
            imwrite(out_file, hough);
            if(threshold==75 && step_theta==0)
                appendImgToCanvas(hough);
            // appendImgToCanvas(hough);
            ulines.clear();
        }
    }

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
        Size s(canvas.cols + img.cols + 5, canvas.rows > img.rows ? canvas.rows : img.rows);
        size_t old_w = canvas.cols;
        copyMakeBorder(canvas, canvas, 0, 0, 0, img.cols + 5, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
        img.copyTo(canvas(Rect(old_w + 5, 0, img.cols, img.rows)));
    }
}
