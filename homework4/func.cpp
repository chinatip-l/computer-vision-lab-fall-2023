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

vector<ContourPoint> initContourPointCircle(int cx, int cy, int r, int cnt)
{
    vector<ContourPoint> tmp;
    float theta = 0, d_theta = (2 * CV_PI) / cnt;
    for (int i = 0; i < cnt; i++)
    {
        float x, y;
        x = (r * cosf32(theta)) + cx;
        y = (r * sinf32(theta)) + cy;
        ContourPoint p;
        p.x = x;
        p.y = y;
        p.energy = numeric_limits<float>::max();
        tmp.push_back(p);
        theta += d_theta;
    }
    return tmp;
}

// maybe calculate energy and return, compare outside
void activeContour(Mat magnitude, Mat direction, vector<ContourPoint> &contour, float alpha, float beta, float gamma)
{
    int p_cnt = contour.size();
    vector<ContourPoint> prev_contour(contour);
    for (int i = 0; i < p_cnt; i++)
    {
        ContourPoint pp, p, pn;
        pp = prev_contour[(i - 1 + p_cnt) % p_cnt];
        p = prev_contour[i % p_cnt];
        pn = prev_contour[(i + 1) % p_cnt];
        // p.x += alpha * (p - pp).x + beta * ((pn - (2 * p) + pp)).x;
        // p.y += alpha * (p - pp).y + beta * ((pn - (2 * p) + pp)).y;

        Vec2f img_force(0, 0), tmp;
        int w_size = 45, offset = w_size / 2;
        for (int wj = 0; wj < w_size; wj++)
        {
            for (int wi = 0; wi < w_size; wi++)
            {
                int v_mag = magnitude.at<Vec3b>(wj - offset, wi - offset)[0];
                float v_theta = direction.at<Vec3f>(wj - offset, wi - offset)[0];
                float x = 0, y = 0;
                if (!isnanf(v_theta))
                {
                    // v_theta+=CV_PI;
                    x = v_mag * cosf32(v_theta);
                    y = v_mag * sinf32(v_theta);
                }
                tmp = Vec2f(x, y);
                img_force += tmp;
            }
        }
        img_force /= -w_size * w_size;

        float e_cont, e_curve, e_image, e_point;
        int mag = magnitude.at<Vec3b>((int)p.y, (int)p.x)[0];
        float theta = direction.at<Vec3f>(p.y, p.x)[0];

        // if (!isnanf(theta))
        // {
        //     p.x -= gamma * mag * cos(theta);
        //     p.y -= gamma * mag * sin(theta);
        // }

        if (pn.settle || pp.settle)
        {
            beta = 0;
        }

        p.x += gamma * img_force[0];
        p.y += gamma * img_force[1];
        e_curve = powf32((p - pp).x, 2) + powf32((p - pp).y, 2);
        e_cont = powf32((pn + pp - 2 * p).x, 2) + powf32((pn + pp - 2 * p).y, 2);
        e_image = -mag * mag;
        e_point = (alpha * e_cont) + (beta * e_curve) + (gamma * e_image);

        if (e_point > p.energy && e_point < 0)
        {
            contour[i].settle = true;
            continue;
        }
        printf("%f %f %f %f %f %d\n", e_cont, e_curve, e_image, e_point, p.energy, e_point > p.energy);

        contour[i].x += alpha * (pn + pp - 2 * p).x + beta * ((p - pp)).x + gamma * img_force[0];
        contour[i].y += alpha * (pn + pp - 2 * p).y + beta * ((p - pp)).y + gamma * img_force[1];
        contour[i].energy = e_point;

        // if (!isnanf(theta))
        // {
        //     // contour[i].x -= gamma * mag * cos(theta);
        //     // contour[i].y -= gamma * mag * sin(theta);

        // }
        // contour[i].x += gamma * img_force[0];
        // contour[i].y += gamma * img_force[1];
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

void activeContour2(Mat magnitude, Mat direction, vector<ContourPoint> &contour, float alpha, float beta, float gamma)
{
    int p_cnt = contour.size();
    vector<ContourPoint> prev_contour(contour);
    for (int i = 0; i < p_cnt; i++)
    {
        ContourPoint pp, p, pn;
        pp = prev_contour[(i - 1 + p_cnt) % p_cnt];
        p = prev_contour[i % p_cnt];
        pn = prev_contour[(i + 1) % p_cnt];
        // p.x += alpha * (p - pp).x + beta * ((pn - (2 * p) + pp)).x;
        // p.y += alpha * (p - pp).y + beta * ((pn - (2 * p) + pp)).y;

        // Vec2f img_force(0, 0), tmp;
        // int w_size = 45, offset = w_size / 2;
        // for (int wj = 0; wj < w_size; wj++)
        // {
        //     for (int wi = 0; wi < w_size; wi++)
        //     {
        //         int v_mag = magnitude.at<Vec3b>(wj - offset, wi - offset)[0];
        //         float v_theta = direction.at<Vec3f>(wj - offset, wi - offset)[0];
        //         float x = 0, y = 0;
        //         if (!isnanf(v_theta))
        //         {
        //             // v_theta+=CV_PI;
        //             x = v_mag * cosf32(v_theta);
        //             y = v_mag * sinf32(v_theta);
        //         }
        //         tmp = Vec2f(x, y);
        //         img_force += tmp;
        //     }
        // }
        // img_force /= -w_size * w_size;

        float e_cont, e_curve, e_image, e_point;

        float theta = direction.at<Vec3f>(p.y, p.x)[0];

        // if (pn.settle || pp.settle)
        // {
        //     beta = 0;
        // }
        // if(pp.settle){
        //     alpha=alpha*2;
        // }

        int s_size = 25, s_offset = s_size / 2;
        int min_i = 0, min_j = 0;
        float min_e_point = numeric_limits<float>::max();
        for (int wj = 0; wj < s_size; wj++)
        {
            for (int wi = 0; wi < s_size; wi++)
            {
                ContourPoint tmp;

                tmp.x = p.x - s_offset + wi;
                tmp.y = p.y - s_offset + wj;
                int tmp_mag = magnitude.at<Vec3b>((int)tmp.y, (int)tmp.x)[0];
                
                e_cont = powf32((pn + pp - 2 * tmp).x, 2) + powf32((pn + pp - 2 * tmp).y, 2);
                e_curve = powf32((tmp - pp).x, 2) + powf32((tmp - pp).y, 2);
                e_image = -tmp_mag * tmp_mag;
                e_point = (alpha * e_cont) + (beta * e_curve) + (gamma * e_image);
                if (e_point < min_e_point)
                {
                    min_e_point = e_point;
                    min_i = tmp.x;
                    min_j = tmp.y;
                }
            }
        }
        contour[i].x = min_i;
        contour[i].y = min_j;
        contour[i].energy=min_e_point;
        // if (min_e_point < 0)
        // {
        //     contour[i].settle = true;
        //     continue;
        // }

        // p.x += gamma * img_force[0];
        // p.y += gamma * img_force[1];
        // e_curve = powf32((p - pp).x, 2) + powf32((p - pp).y, 2);
        // e_cont = powf32((pn + pp - 2 * p).x, 2) + powf32((pn + pp - 2 * p).y, 2);
        // e_image = -mag * mag;
        // e_point = (alpha * e_cont) + (beta * e_curve) + (gamma * e_image);

        // printf("%f %f %f %f %f %d\n", e_cont, e_curve, e_image, e_point, p.energy, e_point > p.energy);

        // contour[i].x += alpha * (pn + pp - 2 * p).x + beta * ((p - pp)).x + gamma * img_force[0];
        // contour[i].y += alpha * (pn + pp - 2 * p).y + beta * ((p - pp)).y + gamma * img_force[1];
        // contour[i].energy = e_point;

        // if (!isnanf(theta))
        // {
        //     // contour[i].x -= gamma * mag * cos(theta);
        //     // contour[i].y -= gamma * mag * sin(theta);

        // }
        // contour[i].x += gamma * img_force[0];
        // contour[i].y += gamma * img_force[1];
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

float contourEnergy(vector<ContourPoint> &contour)
{
    float tmp = 0;
    int cnt = contour.size();
    for (int c = 0; c < cnt; c++)
    {
        if (!isinff(contour[c].energy) && contour[c].energy!=numeric_limits<float>::max())
        {
            tmp += contour[c].energy;
        }
    }
    return tmp;
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

void showGradient(Mat &input, Mat magnitude, Mat direction, int grid)
{
    int r = 5;
    for (int j = 0; j < input.rows; j += grid)
    {
        for (int i = 0; i < input.cols; i += grid)
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
