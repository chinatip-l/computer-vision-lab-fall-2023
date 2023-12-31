#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <string.h>
#include "include/func.hpp"
#include <opencv2/videoio/videoio_c.h>

#define MAX_LEN 250

// path for lab
#define BASE_PATH "/home/chinatip/computer-vision-lab-fall-2023/homework4"
#define SRC_PATH "test_img"
#define OUT_PATH "output"

// define file name, can uncomment to select the input
// #define FILENAME "img1"
// #define FILENAME "img2"
#define FILENAME "img3"

#define EXT "jpg"

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

    Mat gaussian_filter = createGaussianFilter(3, 1);
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

    Mat direction;
    direction = calculateEdgeDirection(sobelx, sobely);

    Mat mag2, dir2, dx, dy, res;
    dx = applySobelX(magnitude);
    dy = applySobelY(magnitude);
    mag2 = calculateEdgeStrength(dx, dy);
    dir2 = calculateEdgeDirection(dx, dy);

    vector<ContourPoint> contour;

    VideoWriter video;
    char video_name[100]="\0";
    snprintf(video_name, 99, "%s/%s/vid_%s.mp4", BASE_PATH, OUT_PATH, FILENAME);
    video.open(video_name , CV_FOURCC('m','p','4','v'), 10,og_img.size(), true);

    contour = initContourPointCircle(magnitude.cols / 2, magnitude.rows / 2, 400, 200);
    float alpha = 0.05, beta = 0.00001, gamma = 0.5;
    float min_e = numeric_limits<float>::max(), current_e, e_1 = 0, e_2 = 1, e_3 = -1, e_4 = 0;
    int osc_cnt = 0;
    vector<Mat> buf;
    char iter_text[25]="\0";
    for (int k = 0; k < 10000; k++)
    {
        Mat i;
        // activeContour(mag2,dir2,contour,alpha,beta,gamma);
        activeContour2(mag2, dir2, contour, alpha, beta, gamma);
        current_e = contourEnergy(contour);
        // activeContour2(magnitude,direction,contour,alpha,beta,gamma);

        Mat mag_with_field;
        og_img.copyTo(mag_with_field);
        // showGradient(mag_with_field,magnitude,direction,2);
        i = showSnake(mag_with_field, contour);
        e_4 = e_3;
        e_3 = e_2;
        e_2 = e_1;
        e_1 = current_e;
        double diff1 = e_2 - e_1;
        double diff2 = e_3 - e_2;
        double diff3 = e_4 - e_3;
        bool isOscillating = (diff1 * diff2 < 0) && (diff2 * diff3 < 0);
        bool isConverging = std::abs(diff1) == std::abs(diff2) && std::abs(diff2) == std::abs(diff3);

        if (!(isOscillating && isConverging))
        {
            min_e = current_e;
            osc_cnt = 0;
        }
        else
        {
            osc_cnt += 1;
            if (osc_cnt == 20)
            {
                
                snprintf(iter_text, 24, "Iteration: %d", k);
                putText(i,iter_text,Point2i(10,30),FONT_HERSHEY_SIMPLEX,1,(0,0,0),2);
                printf("-frame %d %f\n", k, min_e);
                snprintf(out_file, MAX_LEN, "%s/%s/result_%s.%s", BASE_PATH, OUT_PATH, FILENAME, EXT);
                imwrite(out_file,i);
                imshow("Original", i);
                video.write(i);
                video.release();
                waitKey(10);
                break;
            }
        }

        if(k%10==0){
            snprintf(iter_text, 24, "Iteration: %d", k);
            putText(i,iter_text,Point2i(10,30),FONT_HERSHEY_SIMPLEX,1,(0,0,0),2);
            printf("frame %d %f\n", k, min_e);
            imshow("Original", i);
            video.write(i);
            waitKey(10);
        }
        
    }
    printf("Finish\n");
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
