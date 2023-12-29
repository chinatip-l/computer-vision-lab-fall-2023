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
                        0, 0, 0,
                        1, 2, 1};
    Mat sobelY(3, 3, CV_32F, c_sobelY);
    Mat result = applyFilter(img_in, sobelY, 1, true);
    return result;
}

Mat calculateEdgeStrength(Mat x, Mat y)
{
    Mat strength = Mat::zeros(x.rows, x.cols, CV_8UC3);
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
                tmp = round(sqrtf32(tmp));
                if (tmp > 255)
                {
                    tmp = 255;
                }
                strength.at<Vec3b>(j, i)[k] = (int)tmp;
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
                // direction.at<Vec3f>(j, i)[k] = atan2f32(y.at<Vec3f>(j, i)[k], x.at<Vec3f>(j, i)[k]) * 10;

                // if (y.at<Vec3i>(j, i)[k] == 0 && x.at<Vec3i>(j, i)[k] == 0)
                // {
                //     direction.at<Vec3f>(j, i)[k] = 0;
                // }
                // else
                // {
                //     direction.at<Vec3f>(j, i)[k] = atan2f32(y.at<Vec3i>(j, i)[k], x.at<Vec3i>(j, i)[k]);
                // }
                direction.at<Vec3f>(j, i)[k] = atan2f32(y.at<Vec3i>(j, i)[k], x.at<Vec3i>(j, i)[k]);
            }
        }
    }
    return direction;
}

vector<Point2f> initContourPointCircle(int cx, int cy, int r, int cnt)
{
    vector<Point2f> tmp;
    float theta = 0, d_theta = (2 * CV_PI) / cnt;
    for (int i = 0; i < cnt; i++)
    {
        float x, y;
        x = (r * cosf32(theta)) + cx;
        y = (r * sinf32(theta)) + cy;
        tmp.push_back(Point2f(x, y));
        theta += d_theta;
    }
    return tmp;
}

// maybe calculate energy and return, compare outside
void activeContour(Mat magnitude, Mat direction, vector<Point2f> &contour, float alpha, float beta, float gamma)
{
    int p_cnt = contour.size();
    vector<Point2f> prev_contour(contour);
    for (int i = 0; i < p_cnt; i++)
    {
        Point2f pp, p, pn;
        pp = prev_contour[(i - 1 + p_cnt) % p_cnt];
        p = prev_contour[i % p_cnt];
        pn = prev_contour[(i + 1) % p_cnt];
        float e_cont, e_curve, e_image;
        e_cont = powf32((p - pp).x, 2) + powf32((p - pp).y, 2);
        e_curve = powf32((pn - (2 * p) + pp).x, 2) + powf32((pn - (2 * p) + pp).y, 2);
        // e_cont=powf32((p-pp).x,2)+powf32((p-pp).y,2);
        // printf("%f\n", e_curve);
        // contour[i].x += alpha * (pn - p).x + beta * ((pn - p) - (p - pp)).x;
        // contour[i].y += alpha * (pn - p).y + beta * ((pn - p) - (p - pp)).y;
        contour[i].x += alpha* (pn - p).x + beta * ((pn - pp)).x;
        contour[i].y += alpha * (pn - p).y + beta * ((pn - pp)).y;
        // printf("    point #%d %f,%f\n",(i-1+p_cnt)%p_cnt,(pp).x,(pp).y);
        // contour[i].x-=image.at<Vec3b>(p.y,p.x)[0]>0?image.at<Vec3b>(p.y,p.x)[0]:1;
        // contour[i].y-=image.at<Vec3b>(p.y,p.x)[0]>0?image.at<Vec3b>(p.y,p.x)[0]:1;

        // printf("ret t %f %f\n",p.x,p.y);
        float mag = magnitude.at<Vec3b>((int)p.y, (int)p.x)[0];
        // if (mag < 1)
        // {
        //     mag = 1;
        // }
        // printf("ret r\n");
        float theta = direction.at<Vec3f>(p.y, p.x)[0];
        if (!isnanf(theta))
        {
            contour[i].x += gamma * mag * cos(theta);
            contour[i].y += gamma * mag * sin(theta);
        }
        // printf("%d - %f\n",i, theta);

        if (contour[i].x < 0 || contour[i].x != contour[i].x)
        {
            contour[i].x = 0;
        }
        if (contour[i].x >= magnitude.cols || contour[i].x != contour[i].x)
        {
            contour[i].x = magnitude.cols - 1;
        }
        if (contour[i].y < 0 || contour[i].y != contour[i].y)
        {
            contour[i].y = 0;
        }
        if (contour[i].y >= magnitude.rows || contour[i].y != contour[i].y)
        {
            contour[i].y = magnitude.rows - 1;
        }
    }
}

Mat showSnake(Mat input, vector<Point2f> contour)
{
    Mat res = input.clone();
    int p_cnt = contour.size();
    for (int i = 0; i < p_cnt; i++)
    {
        Point2f p = contour[i % p_cnt];
        Point2f pn = contour[(i + 1) % p_cnt];
        circle(res, p, 3, (255, 255, 128, 128), -1);
        circle(res, pn, 3, (255, 255, 128), -1);
        line(res, p, pn, (255, 255, 128), 1);
    }
    return res;
}
