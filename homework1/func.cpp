#include "include/func.hpp"

Mat applyGreyscaleFilter(Mat img)
{
    Mat bw_img(img.size(), img.type());
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            Vec3b brg_px = img.at<Vec3b>(i, j);
            int b = (brg_px[0] + brg_px[1] + brg_px[2]) / 3;
            brg_px[0] = b;
            brg_px[1] = b;
            brg_px[2] = b;
            bw_img.at<Vec3b>(i, j) = brg_px;
        }
    }
    return bw_img;
}

Mat applyConvolve(Mat img, Mat kernel)
{
    int padding = 1, stride = 1;
    return applyConvolve(img, kernel, padding, stride);
}

Mat applyConvolve(Mat img, Mat kernel, int padding, int stride)
{

    int ker_x_offset = kernel.cols / 2;
    int ker_y_offset = kernel.rows / 2;
    int ker_w = kernel.cols;
    int ker_h = kernel.rows;
    int out_w = ((img.cols + (2 * padding) - kernel.cols) / stride) + 1;
    int out_h = ((img.rows + (2 * padding) - kernel.rows) / stride) + 1;

    Mat padded;
    copyMakeBorder(img, padded, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

    Mat img_res = Mat::zeros(out_h, out_w, CV_32SC3);
    Vec3i res(0, 0, 0);
    for (int img_j = 1; img_j <= img.rows; img_j += stride)
    {
        for (int img_i = 1; img_i <= img.cols; img_i += stride)
        {

            Mat windowed = padded(Rect(img_i - ker_x_offset, img_j - ker_y_offset, ker_w, ker_h));
            res = Vec3i(0, 0, 0);
            for (int ker_j = 0; ker_j < kernel.rows; ker_j++)
            {
                for (int ker_i = 0; ker_i < kernel.cols; ker_i++)
                {
                    Vec3b t = windowed.at<Vec3b>(ker_i, ker_j);
                    int k = kernel.at<int>(ker_i, ker_j);
                    for (int c = 0; c < 3; c++)
                    {
                        res[c] += t[c] * k;
                    }
                }
            }

            img_res.at<Vec3i>((img_j - 1) / stride, (img_i - 1) / stride) = res;
        }
    }

    return img_res;
}

Mat applyMaxPooling(Mat img, int kernel_w, int kernel_h, int stride)
{
    int pad_w = 0, pad_h = 0;
    if (img.cols % kernel_w > 0)
    {
        pad_w = kernel_w - (img.cols % kernel_w);
    }
    if (img.rows % kernel_h > 0)
    {
        pad_h = kernel_h - (img.rows % kernel_h);
    }

    Mat padded;
    copyMakeBorder(img, padded, 0, pad_h, 0, pad_w, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
    int out_w = ((padded.cols - kernel_w) / stride) + 1;
    int out_h = ((padded.rows - kernel_h) / stride) + 1;
    Mat img_res = Mat::zeros(out_h, out_w, CV_32SC3);

    Vec3i res(0, 0, 0);
    for (int img_j = 0; img_j < img_res.rows; img_j += 1)
    {
        for (int img_i = 0; img_i < img_res.cols; img_i += 1)
        {
            Mat windowed = padded(Rect(img_i*stride, img_j*stride, kernel_w, kernel_h));
            res = Vec3i(0, 0, 0);
            for (int ker_j = 0; ker_j < kernel_h; ker_j++)
            {

                for (int ker_i = 0; ker_i < kernel_w; ker_i++)
                {
                    

                    Vec3i t = windowed.at<Vec3i>(ker_i, ker_j);
                    if (t[0] > res[0])
                    {
                        res[0] = t[0];
                    }
                    if (t[1] > res[1])
                    {
                        res[1] = t[1];
                    }
                    if (t[2] > res[2])
                    {
                        res[2] = t[2];
                    }
                }
            }
            img_res.at<Vec3i>(img_j, img_i) = res;
            
            
        }
    }
    return img_res;
}

Mat applyBinarisation(Mat img, int thres)
{

    Mat img_res = Mat::zeros(img.rows, img.cols, CV_32SC3);
    Vec3i res = Vec3i(0);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            res = Vec3i(0, 0, 0);
            Vec3i bgrPixel = img.at<Vec3i>(i, j);
            for (int c = 0; c < 3; c++)
            {
                res[c] = bgrPixel[c] > thres ? 255 : 0;
            }
            img_res.at<Vec3i>(i, j) = res;
        }
    }
    return img_res;
}