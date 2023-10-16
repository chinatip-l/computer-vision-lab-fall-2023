#include "include/func.hpp"

Mat applyGreyscaleFilter(Mat img)
{
    // Create New Image as Output Image
    Mat bw_img(img.size(), img.type());
    for (int j = 0; j < img.rows; j++)
    {
        for (int i = 0; i < img.cols; i++)
        {
            // Get the pixel value from the original image
            Vec3b brg_px = img.at<Vec3b>(j, i);

            // Normal average function ie. sum(xn)/n ;
            int b = (brg_px[0] + brg_px[1] + brg_px[2]) / 3;

            // Write back to every channel
            brg_px[0] = b;
            brg_px[1] = b;
            brg_px[2] = b;

            // Write to new image buffer
            bw_img.at<Vec3b>(j, i) = brg_px;
        }
    }
    // return the output image
    return bw_img;
}

Mat applyConvolve(Mat img, Mat kernel)
{
    // apply default padding and stride
    int padding = 1, stride = 1;
    return applyConvolve(img, kernel, padding, stride);
}

Mat applyConvolve(Mat img, Mat kernel, int padding, int stride)
{
    // calculate kernel offset ie. number of pixel before and 
    // after since the target pixel will be the center of 
    // matrix,
    // in order to get the neighbour pixel with size of kernel
    // size, offset will be applied for both before, and after
    // the target pixel
    // example
    // width = 5 => offset = (int)5/2 = 2;
    // px   ... k   n-offset    ... n   ... n+offset    k
    int ker_x_offset = kernel.cols / 2;
    int ker_y_offset = kernel.rows / 2;

    // get kernel size
    int ker_w = kernel.cols;
    int ker_h = kernel.rows;

    // calculate output image size
    int out_w = ((img.cols + (2 * padding) - kernel.cols) / stride) + 1;
    int out_h = ((img.rows + (2 * padding) - kernel.rows) / stride) + 1;

    // create intermediate buffer and add padding, size is 
    // w+(2*padding),h+(2*padding) then copy the content 
    // of image to the buffer
    Mat padded;
    padded = Mat::zeros(img.rows + (2 * padding), img.cols + (2 * padding), CV_32SC3);
    for (int j = 0; j < img.rows; j++)
    {
        for (int i = 0; i < img.cols; i++)
        {
            padded.at<Vec3i>(j+padding,i+padding)=img.at<Vec3b>(j,i);
        }
    }
    
    // create output buffer with calculated output size
    // data type need to be vector of signed int 32 bit
    // in case negative data is possible, this can help
    // preserve information through the convolution process
    // and rasterise to 8 bit at the last step
    Mat img_res = Mat::zeros(out_h, out_w, CV_32SC3);
    Vec3i res(0, 0, 0);

    // iterate over the size of input image
    for (int img_j = 1; img_j <= img.rows; img_j += stride)
    {
        for (int img_i = 1; img_i <= img.cols; img_i += stride)
        {

            // get windowed matrix from the padded buffer with 
            // the size of kernel size, and target pixel is in
            // the center of windowed matrix
            Mat windowed = padded(Rect(img_i - ker_x_offset, img_j - ker_y_offset, ker_w, ker_h));
            res = Vec3i(0, 0, 0);

            // item-wise multiply windowed matrix with kernel 
            // matrix and summarise 
            for (int ker_j = 0; ker_j < kernel.rows; ker_j++)
            {
                for (int ker_i = 0; ker_i < kernel.cols; ker_i++)
                {
                    Vec3b t = windowed.at<Vec3i>(ker_i, ker_j);
                    int k = kernel.at<int>(ker_i, ker_j);
                    for (int c = 0; c < 3; c++)
                    {
                        res[c] += t[c] * k;
                    }
                }
            }

            // write to output buffer
            img_res.at<Vec3i>((img_j - 1) / stride, (img_i - 1) / stride) = res;
        }
    }

    // return the output image
    return img_res;
}

Mat applyMaxPooling(Mat img, int kernel_w, int kernel_h, int stride)
{
    // calculate number of row and column to be padded by
    // n_fill = ker_size - (img_size % ker_size)
    // if img_size % ker_size != 0
    int pad_w = 0, pad_h = 0;
    if (img.cols % kernel_w > 0)
    {
        pad_w = kernel_w - (img.cols % kernel_w);
    }
    if (img.rows % kernel_h > 0)
    {
        pad_h = kernel_h - (img.rows % kernel_h);
    }

    // create padded buffer, append padding to the last row/column
    Mat padded;
    padded = Mat::zeros(img.rows + pad_h, img.cols + pad_w, CV_32SC3);
    for (int j = 0; j < img.rows; j++)
    {
        for (int i = 0; i < img.cols; i++)
        {
            padded.at<Vec3i>(j,i)=img.at<Vec3i>(j,i);
        }
    }

    // calculate output size
    int out_w = ((padded.cols - kernel_w) / stride) + 1;
    int out_h = ((padded.rows - kernel_h) / stride) + 1;

    // create output buffer with signed int 32 bit data
    // the reason is as mentioned before
    Mat img_res = Mat::zeros(out_h, out_w, CV_32SC3);
    Vec3i res(0, 0, 0);

    // iterate over the size of output image, the location
    // of output can calculate back to the range if input pixel
    for (int img_j = 0; img_j < img_res.rows; img_j += 1)
    {
        for (int img_i = 0; img_i < img_res.cols; img_i += 1)
        {

            // get the windowed matrix by calculate the input
            // starting point from output location and size 
            // of kernel
            Mat windowed = padded(Rect(img_i * stride, img_j * stride, kernel_w, kernel_h));
            res = Vec3i(0, 0, 0);

            // iterate over each pixel in windowed matrix and 
            // compare to find the maximum value in window,
            // store it as a representative of the window
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

            // store the maximum value to buffer
            img_res.at<Vec3i>(img_j, img_i) = res;
        }
    }

    // return the output image
    return img_res;
}

Mat applyBinarisation(Mat img, int thres)
{

    // create output buffer, the size is the same as input
    Mat img_res = Mat::zeros(img.rows, img.cols, CV_32SC3);
    Vec3i res = Vec3i(0);

    // iterate over each pixel and compare to threshold, 
    // if greater than threshold, return 255, else return 0
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

            // store to buffer
            img_res.at<Vec3i>(i, j) = res;
        }
    }

    // return the output image
    return img_res;
}