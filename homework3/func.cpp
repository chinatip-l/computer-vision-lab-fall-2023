#include "include/func.hpp"
#include <vector>
#include <cmath>
const double K_E = std::exp(1.0);

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

float calculateGaussian(int x, int y, float sigma)
{
    return (1 / (CV_2PI * sigma * sigma)) * (powf32(K_E, -1 * ((x * x) + (y * y)) / (2 * sigma * sigma)));
}

Mat createGaussianFilter(int size, float sigma)
{
    int center = size / 2;
    Mat kernel = Mat::zeros(size, size, CV_32F);
    for (int j = 0; j < kernel.rows; j++)
    {
        for (int i = 0; i < kernel.cols; i++)
        {
            kernel.at<float>(j, i) = calculateGaussian(i - center, j - center, sigma);
        }
    }
    return kernel;
}

Mat applyFilter(Mat img, Mat kernel, int stride, bool suppress_border)
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

    int padding = ker_x_offset;

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
    int dtype = img.type();

    // here, we convert data type while copy to make it compatible
    // for higher precision computation
    // check if input is uint8 or int 32
    // if uint8 then we will channel-wise copy to padded buffer
    // due to different data alignment
    // if int38 then we will pixel-wise copy to padded buffer
    switch (dtype)
    {
    case CV_32SC3:
        for (int j = 0; j < img.rows; j++)
        {
            for (int i = 0; i < img.cols; i++)
            {
                padded.at<Vec3i>(j + padding, i + padding) = img.at<Vec3i>(j, i);
            }
        }
        break;
    case CV_8UC3:
        for (int j = 0; j < img.rows; j++)
        {
            for (int i = 0; i < img.cols; i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    padded.at<Vec3i>(j + padding, i + padding)[k] = img.at<Vec3b>(j, i)[k];
                }
            }
        }
        break;
    default:
        break;
    }

    // create output buffer with calculated output size
    // data type need to be vector of signed int 32 bit
    // in case negative data is possible, this can help
    // preserve information through the convolution process
    // and rasterise to 8 bit at the last step
    Mat img_res = Mat::zeros(out_h, out_w, CV_32SC3);

    Vec3i res(0, 0, 0);

    // iterate over the size of input image
    for (int img_j = 0; img_j < img.rows; img_j += stride)
    {
        for (int img_i = 0; img_i < img.cols; img_i += stride)
        {
            // check if it is a border and suppress border if selected
            if (img_i < ker_w || img_j < ker_h || img_i > img.cols - ker_h - 1 || img_j > img.rows - ker_h - 1)
            {
                if (suppress_border)
                {
                    res = Vec3i(0, 0, 0);
                    img_res.at<Vec3i>((img_j) / stride, (img_i) / stride) = res;
                    continue;
                }
            }

            // get windowed matrix from the padded buffer with
            // the size of kernel size, and target pixel is in
            // the center of windowed matrix
            Mat windowed = padded(Rect(img_i, img_j, ker_w, ker_h));
            res = Vec3i(0, 0, 0);
            float tmp = 0;
            float tmp1 = 0;
            float tmp2 = 0;
            float tmp3 = 0;

            // item-wise multiply windowed matrix with kernel
            // matrix and summarise
            for (int ker_j = 0; ker_j < kernel.rows; ker_j++)
            {
                for (int ker_i = 0; ker_i < kernel.cols; ker_i++)
                {
                    Vec3b t = windowed.at<Vec3i>(ker_j, ker_i);
                    float k = kernel.at<float>(ker_j, ker_i);
                    tmp += (t[0] * k);
                    tmp1 += (t[0] * k);
                    tmp2 += (t[1] * k);
                    tmp3 += (t[2] * k);
                }
            }
            res[0] = tmp1;
            res[1] = tmp2;
            res[2] = tmp3;

            // write to output buffer
            img_res.at<Vec3i>((img_j) / stride, (img_i) / stride) = res;
        }
    }
    // return the output image
    return img_res;
}

Mat applySobelX(Mat img_in)
{
    float c_sobelX[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
    Mat sobelX(3, 3, CV_32F, c_sobelX);
    Mat result = applyFilter(img_in, sobelX, 1, true);
    return result;
}

Mat applySobelY(Mat img_in)
{
    float c_sobelY[] = {-1, -2, -1,
                         0,  0,  0,
                         1,  2,  1};
    Mat sobelY(3, 3, CV_32F, c_sobelY);
    Mat result = applyFilter(img_in, sobelY, 1, true);
    return result;
}

Mat calculateEdgeStrength(Mat x, Mat y)
{
    Mat strength = Mat::zeros(x.rows, x.cols, CV_32FC3);
    float tmp = 0;
    for (int j = 0; j < strength.rows; j += 1)
    {
        for (int i = 0; i < strength.cols; i += 1)
        {
            for (int k = 0; k < 3; k++)
            {
                // Strength of pixel is calculated by
                // sqrt(gx^2 + gy^2)
                tmp = (x.at<Vec3i>(j, i)[k] * x.at<Vec3i>(j, i)[k]) + (y.at<Vec3i>(j, i)[k] * y.at<Vec3i>(j, i)[k]);
                strength.at<Vec3f>(j, i)[k] = sqrtf32(tmp);
            }
        }
    }

    return strength;
}

Mat calculateEdgeDirection(Mat x, Mat y)
{
    Mat direction = Mat::zeros(x.rows, x.cols, CV_32FC3);
    for (int j = 0; j < direction.rows; j += 1)
    {
        for (int i = 0; i < direction.cols; i += 1)
        {
            for (int k = 0; k < 3; k++)
            {
                // direction of pixel is calculated by
                // atan(y/x)
                direction.at<Vec3f>(j, i)[k] = atan2f32(y.at<Vec3f>(j, i)[k], x.at<Vec3f>(j, i)[k]) * 10;
            }
        }
    }
    return direction;
}

int quantiseDirection(float angle)
{
    // convert radiant to degree
    angle = angle * 180 / CV_PI;

    // quantise to 4 axes 
    if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180))
    {
        return 0;
    }
    else if (angle >= 22.5 && angle < 67.5)
    {
        return 45;
    }
    else if (angle >= 67.5 && angle < 112.5)
    {
        return 90;
    }
    else if (angle >= 112.5 && angle < 157.5)
    {
        return 135;
    }
    else
    {
        return 0;
    }
}

Mat calculateNonMaximumSuppression(cv::Mat strength, cv::Mat direction)
{
    // Create a zero-initialized matrix with the same size as the input, to store the result
    Mat res = Mat::zeros(strength.rows, strength.cols, CV_32SC3);

    // Iterate over the image, avoiding the borders (as the algorithm needs neighboring pixels)
    for (int j = 1; j < strength.rows - 1; j++)
    {
        for (int i = 1; i < strength.cols - 1; i++)
        {
            // Iterate over each channel (assuming a 3-channel image)
            for (int k = 0; k < 3; k++)
            {
                // Quantize the direction to one of the four possible angles (0, 45, 90, 135)
                int dir = quantiseDirection(direction.at<Vec3f>(j, i)[k]);

                // Retrieve the strength of the neighboring pixels in all 8 directions
                float neighbour_UL = strength.at<Vec3f>(j - 1, i - 1)[k];
                float neighbour_UM = strength.at<Vec3f>(j - 1, i)[k];
                float neighbour_UR = strength.at<Vec3f>(j - 1, i + 1)[k];
                float neighbour_ML = strength.at<Vec3f>(j, i - 1)[k];
                float neighbour_CC = strength.at<Vec3f>(j, i)[k]; // Current pixel
                float neighbour_MR = strength.at<Vec3f>(j, i + 1)[k];
                float neighbour_DL = strength.at<Vec3f>(j + 1, i - 1)[k];
                float neighbour_DM = strength.at<Vec3f>(j + 1, i)[k];
                float neighbour_DR = strength.at<Vec3f>(j + 1, i + 1)[k];

                // Initially set the result pixel to the current strength
                res.at<Vec3i>(j, i)[k] = (int)neighbour_CC;

                // Depending on the quantized direction, suppress non-maximum pixels
                // by setting them to zero if they are not the local maxima along the gradient direction
                switch (dir)
                {
                case 0:
                    if (neighbour_CC <= neighbour_ML || neighbour_CC <= neighbour_MR)
                        res.at<Vec3f>(j, i)[k] = 0;
                    break;
                case 45:
                    if (neighbour_CC <= neighbour_UR || neighbour_CC <= neighbour_DL)
                        res.at<Vec3f>(j, i)[k] = 0;
                    break;
                case 90:
                    if (neighbour_CC <= neighbour_UM || neighbour_CC <= neighbour_DM)
                        res.at<Vec3f>(j, i)[k] = 0;
                    break;
                case 135:
                    if (neighbour_CC <= neighbour_UL || neighbour_CC <= neighbour_DR)
                        res.at<Vec3f>(j, i)[k] = 0;
                    break;
                }
            }
        }
    }

    // Return the resulting image after non-maximum suppression
    return res;
}

Mat applyDoubleThresholding(Mat input, int thres_low, int thres_high)
{
    // Create a zero-initialized matrix with the same size as the input, to store the result
    Mat res = Mat::zeros(input.rows, input.cols, CV_8UC3);

    // Temporary variable to store the intensity of the current pixel
    int tmp = 0;

    // Iterate over the image, avoiding the borders
    for (int j = 1; j < input.rows - 1; j++)
    {
        for (int i = 1; i < input.cols - 1; i++)
        {
            // Iterate over each channel of the pixel
            for (int k = 0; k < 3; k++)
            {
                // Get the intensity value of the current pixel
                tmp = input.at<Vec3i>(j, i)[k];

                // If the intensity is higher than the high threshold, mark it as a strong edge (255)
                if (tmp >= thres_high)
                {
                    res.at<Vec3b>(j, i)[k] = 255;
                }
                // If the intensity is lower than the low threshold, suppress it (0)
                else if (tmp < thres_low)
                {
                    res.at<Vec3b>(j, i)[k] = 0;
                }
                // If the intensity is between the low and high thresholds, mark it as a weak edge (128)
                else
                {
                    res.at<Vec3b>(j, i)[k] = 128;
                }
            }
        }
    }

    // Return the resulting image after double thresholding
    return res;
}

Mat applyHysteresis(Mat input)
{
    // Clone the input image to preserve the original and work on a copy
    Mat res = input.clone();

    // Define constants for weak and strong edge values
    int WEAK = 128, STRONG = 255;

    // Iterate over the image, avoiding the borders
    for (int j = 1; j < res.rows - 1; j++)
    {
        for (int i = 1; i < res.cols - 1; i++)
        {
            // Iterate over each channel of the pixel
            for (int k = 0; k < 3; k++)
            {
                // Check if the current pixel is a weak edge
                if (res.at<Vec3b>(j, i)[k] == WEAK)
                {
                    // Initialize a flag to check if the weak edge is connected to a strong edge
                    bool connected = false;

                    // Check the 8 neighboring pixels to see if any of them is a strong edge
                    for (int dj = -1; dj <= 1; ++dj)
                    {
                        for (int di = -1; di <= 1; ++di)
                        {
                            // Skip the current pixel
                            if (di == 0 && dj == 0)
                                continue;

                            // If a strong edge is found in the neighborhood, set the flag
                            if (res.at<Vec3b>(j + dj, i + di)[k] == STRONG)
                            {
                                connected = true;
                                break;
                            }
                        }
                        if (connected)
                            break;
                    }

                    // If the weak edge is connected to a strong edge, strengthen it; otherwise, suppress it
                    res.at<Vec3b>(j, i)[k] = connected ? STRONG : 0;
                }
            }
        }
    }

    // Return the resulting image after applying hysteresis
    return res;
}


void applyHoughTransform(cv::Mat edges, std::vector<cv::Vec2f> &lines, float rho, float theta, int threshold)
{
    // Get the dimensions of the edges image
    int width = edges.cols;
    int height = edges.rows;

    // Calculate the maximum distance (rho) possible in the image
    int rho_max = (int)(std::sqrt(width * width + height * height));

    // Calculate the maximum number of bins for theta
    int theta_max = (int)(CV_PI / theta);

    // Create the accumulator matrix initialized to zero
    cv::Mat accum = cv::Mat::zeros(2 * rho_max, theta_max, CV_32S);

    // Iterate over the pixels of the edges image
    for (int y = 0; y < edges.rows; y++)
    {
        for (int x = 0; x < edges.cols; x++)
        {
            // Check if the current pixel is part of an edge
            if (edges.at<Vec3b>(y, x)[0] > 0)
            {
                // For each angle theta, calculate the corresponding rho value
                for (int t = 0; t < theta_max; t++)
                {
                    double current_theta = (t * theta);

                    // Calculate rho for the current (x, y) point and theta
                    double rho_val = cvRound((x * std::cos(current_theta)) + (y * std::sin(current_theta)));
                    int rho_index = (int)((rho_val / rho) + rho_max);

                    // Increment the accumulator's bin if the index is valid
                    if (rho_index >= 0 && rho_index < 2 * rho_max)
                    {
                        accum.at<int>(rho_index, t)++;
                    }
                }
            }
        }
    }

    // Iterate over the accumulator to find local maxima (lines)
    for (int r = 0; r < 2 * rho_max; r++)
    {
        for (int t = 0; t < theta_max; t++)
        {
            // If the bin's value is above the threshold, it is considered a line
            if (accum.at<int>(r, t) >= threshold)
            {
                // Convert (rho, theta) from accumulator space to image space and add to the lines vector
                lines.push_back(cv::Vec2f(r * rho - rho_max, t * theta));
            }
        }
    }
}
