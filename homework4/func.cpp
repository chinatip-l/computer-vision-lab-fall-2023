#include "include/func.hpp"
#include <vector>
#include <cmath>
#include <limits>

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
            // check if it is a border and suppress border
            // if selected
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
                        0, 0, 0,
                        1, 2, 1};
    Mat sobelY(3, 3, CV_32F, c_sobelY);
    Mat result = applyFilter(img_in, sobelY, 1, true);
    return result;
}

Mat calculateEdgeStrength(Mat grad_x, Mat grad_y)
{
    Mat strength = Mat::zeros(grad_x.rows, grad_x.cols, CV_8UC3);
    float tmp = 0;
    for (int j = 0; j < strength.rows; j += 1)
    {
        for (int i = 0; i < strength.cols; i += 1)
        {
            for (int k = 0; k < 3; k++)
            {
                // Strength of pixel is calculated by
                // sqrt(gx^2 + gy^2)
                tmp = (grad_x.at<Vec3i>(j, i)[k] * grad_x.at<Vec3i>(j, i)[k]) + (grad_y.at<Vec3i>(j, i)[k] * grad_y.at<Vec3i>(j, i)[k]);
                tmp = round(sqrtf32(tmp));

                // add clipping criteria in case of over 255
                // and lower bound for suppressing soft area
                if (tmp > 255)
                {
                    tmp = 255;
                }
                if (tmp < 50)
                {
                    tmp = 0;
                }
                strength.at<Vec3b>(j, i)[k] = (int)tmp;
            }
        }
    }

    return strength;
}

Mat calculateEdgeDirection(Mat grad_x, Mat grad_y)
{
    Mat direction = Mat::zeros(grad_x.rows, grad_x.cols, CV_32FC3);
    for (int j = 0; j < direction.rows; j += 1)
    {
        for (int i = 0; i < direction.cols; i += 1)
        {
            for (int k = 0; k < 3; k++)
            {
                // direction of pixel is calculated by
                // atan(y/x)
                direction.at<Vec3f>(j, i)[k] = atan2f32(grad_y.at<Vec3i>(j, i)[k], grad_x.at<Vec3i>(j, i)[k]);
            }
        }
    }
    return direction;
}

/**
 * @brief Initialise contour points in a circular pattern.
 *
 * This function creates a set of contour points arranged in the shape of a circle.
 * It is useful for initializing contour-based algorithms, especially for circular objects.
 *
 * @param cx The x-coordinate of the circle's center.
 * @param cy The y-coordinate of the circle's center.
 * @param r The radius of the circle.
 * @param quantity The number of contour points to generate.
 * @return vector<ContourPoint> A vector of contour points arranged in a circular pattern.
 */
vector<ContourPoint> initContourPointCircle(int cx, int cy, int r, int quantity)
{
    vector<ContourPoint> tmp;
    float theta = 0, d_theta = (2 * CV_PI) / quantity;

    for (int i = 0; i < quantity; i++)
    {
        float x, y;
        // Calculate the x and y coordinates using polar to 
        // Cartesian conversion
        x = (r * cosf32(theta)) + cx;
        y = (r * sinf32(theta)) + cy;

        // Create a ContourPoint and set its properties
        ContourPoint p;
        p.x = x;
        p.y = y;

        // Since we try to minimise the energy, we initialise 
        // the energy to the maximum float value
        p.energy = numeric_limits<float>::max(); 
        
        // Add the ContourPoint to the vector
        tmp.push_back(p); 
        theta += d_theta;
    }

    // Return the vector of contour points
    return tmp;
}

/**
 * @brief Perform the active contour (snake) algorithm on an image.
 * 
 * This function iteratively adjusts the positions of contour points to align with image features
 * like edges, based on an energy minimization process. The energy terms include continuity,
 * curvature, and image forces.
 * 
 * @param magnitude The magnitude of the gradient of the image.
 * @param direction The direction of the gradient of the image.
 * @param contour A reference to a vector of ContourPoint representing the initial contour.
 * @param alpha The weight for the continuity energy term.
 * @param beta The weight for the curvature energy term.
 * @param gamma The weight for the image energy term.
 */
void activeContour(Mat magnitude, Mat direction, vector<ContourPoint> &contour, float alpha, float beta, float gamma)
{
    int p_cnt = contour.size();
    vector<ContourPoint> prev_contour(contour);

    for (int i = 0; i < p_cnt; i++)
    {
        // Retrieve previous, current, and next points in the contour
        ContourPoint pp, p, pn;
        pp = prev_contour[(i - 1 + p_cnt) % p_cnt];
        p = prev_contour[i % p_cnt];
        pn = prev_contour[(i + 1) % p_cnt];

        // Energy terms
        float e_cont, e_curve, e_image, e_point;

        // Search window settings
        int s_size = 25, s_offset = s_size / 2;
        int min_i = 0, min_j = 0;
        float min_e_point = numeric_limits<float>::max();

        // Search within the window to minimise energy
        for (int wj = 0; wj < s_size; wj++)
        {
            for (int wi = 0; wi < s_size; wi++)
            {
                ContourPoint tmp;

                // Candidate point within the window
                tmp.x = p.x - s_offset + wi;
                tmp.y = p.y - s_offset + wj;
                int tmp_mag = magnitude.at<Vec3b>((int)tmp.y, (int)tmp.x)[0];

                // Calculate energy terms
                e_cont = powf32((pn + pp - 2 * tmp).x, 2) + powf32((pn + pp - 2 * tmp).y, 2);
                e_curve = powf32((tmp - pp).x, 2) + powf32((tmp - pp).y, 2);
                e_image = -tmp_mag * tmp_mag;
                e_point = (alpha * e_cont) + (beta * e_curve) + (gamma * e_image);

                // Update if a lower energy point is found
                if (e_point < min_e_point)
                {
                    min_e_point = e_point;
                    min_i = tmp.x;
                    min_j = tmp.y;
                }
            }
        }

        // Update the contour point with the minimum energy point found
        contour[i].x = min_i;
        contour[i].y = min_j;
        contour[i].energy = min_e_point;
    }
}


/**
 * @brief Calculate the total energy of a contour.
 * 
 * This function computes the sum of the energy values of each point in the contour.
 * It is useful for evaluating the overall energy of the contour, which can be a measure of its
 * alignment with image features or its smoothness, depending on the energy model used.
 * 
 * @param contour A reference to a vector of ContourPoint, each with an associated energy value.
 * @return float The total energy of the contour.
 */
float contourEnergy(vector<ContourPoint> &contour)
{
    float tmp = 0;
    int cnt = contour.size();

    // Iterate over each point in the contour
    for (int c = 0; c < cnt; c++)
    {
        // Check if the energy is not infinity and not the 
        // maximum float value
        // This is done to ensure that only valid, finite 
        // energy values are summed
        if (!isinff(contour[c].energy) && contour[c].energy != numeric_limits<float>::max())
        {
            // Add the point's energy to the total
            tmp += contour[c].energy; 
        }
    }

    return tmp; // Return the total energy of the contour
}


Mat showSnake(Mat input, vector<ContourPoint> contour)
{
    Mat res = input.clone();
    int p_cnt = contour.size();
    for (int i = 0; i < p_cnt; i++)
    {
        ContourPoint p = contour[i % p_cnt];
        ContourPoint pn = contour[(i + 1) % p_cnt];
        circle(res, p, 3, (0, 0, 255), -1);
        circle(res, pn, 3, (0, 0, 255), -1);
        line(res, p, pn, (255, 255, 128), 1);
    }
    return res;
}

void showGradient(Mat &input, Mat magnitude, Mat direction, int stride)
{
    int r = 5;
    for (int j = 0; j < input.rows; j += stride)
    {
        for (int i = 0; i < input.cols; i += stride)
        {
            Point2i c(i, j);
            int mag = magnitude.at<Vec3b>(j, i)[0];
            float theta = direction.at<Vec3f>(j, i)[0];
            if (mag > 0 && !isnanf(theta))
            {
                Point2i e;
                e.x = (int)(r * (mag / 255) * cosf32(theta)) + i;
                e.y = (int)(r * (mag / 255) * sinf32(theta)) + j;
                arrowedLine(input, c, e, (0, 0, mag), 1);
            }
        }
    }
}
