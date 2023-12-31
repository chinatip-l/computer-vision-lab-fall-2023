#ifndef _FUNC_H_
#define _FUNC_H_

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

/**
 * @brief Class representing a point with additional energy attribute.
 *
 * Inherits from OpenCV's Point2f class and adds an energy attribute.
 * This can be used to represent points in an image with an associated energy value.
 */
class ContourPoint : public Point2f
{
public:
    float energy; ///< Energy value of the contour point.
};

/**
 * @brief Applies a greyscale filter to an image.
 *
 * @param image The source image to be converted.
 * @return Grayscale version of the input image.
 */
Mat applyGreyscaleFilter(Mat image);

/**
 * @brief Calculates Gaussian value for given coordinates and sigma.
 *
 * @param x X-coordinate.
 * @param y Y-coordinate.
 * @param sigma Standard deviation of the Gaussian function.
 * @return Gaussian value for the specified coordinates and sigma.
 */
float calculateGaussian(int x, int y, float sigma);

/**
 * @brief Creates a Gaussian filter kernel of a specified size and sigma.
 *
 * @param size Size of the Gaussian filter (typically odd number).
 * @param sigma Standard deviation of the Gaussian.
 * @return Gaussian filter as a Mat object.
 */
Mat createGaussianFilter(int size, float sigma);

/**
 * @brief Applies a filter to an image with optional border suppression.
 *
 * @param image The source image.
 * @param kernel The filter kernel.
 * @param stride Stride for the convolution.
 * @param suppress_border If true, suppresses the border effects.
 * @return The filtered image.
 */
Mat applyFilter(Mat image, Mat kernel, int stride, bool suppress_border);

/**
 * @brief Applies the Sobel operator in the X direction.
 *
 * @param image The source image.
 * @return Image after applying the Sobel X operator.
 */
Mat applySobelX(Mat image);

/**
 * @brief Applies the Sobel operator in the Y direction.
 *
 * @param image The source image.
 * @return Image after applying the Sobel Y operator.
 */
Mat applySobelY(Mat image);

/**
 * @brief Calculates the edge strength from gradient images.
 *
 * @param grad_x Gradient image in the X direction.
 * @param grad_y Gradient image in the Y direction.
 * @return Edge strength image.
 */
Mat calculateEdgeStrength(Mat grad_x, Mat grad_y);

/**
 * @brief Calculates the edge direction from gradient images.
 *
 * @param grad_x Gradient image in the X direction.
 * @param grad_y Gradient image in the Y direction.
 * @return Edge direction image.
 */
Mat calculateEdgeDirection(Mat grad_x, Mat grad_y);

/**
 * @brief Initialises contour points in a circular pattern.
 *
 * @param cx X-coordinate of the circle center.
 * @param cy Y-coordinate of the circle center.
 * @param r Radius of the circle.
 * @param quantity Number of contour points to generate.
 * @return Vector of initialised contour points.
 */
vector<ContourPoint> initContourPointCircle(int cx, int cy, int r, int quantity);

/**
 * @brief Implements the active contour model (snake) algorithm.
 *
 * @param magnitude Magnitude of the gradient.
 * @param direction Direction of the gradient.
 * @param contour Reference to a vector of ContourPoints representing the snake.
 * @param alpha Weight for continuity energy.
 * @param beta Weight for curvature energy.
 * @param gamma Weight for image energy.
 */
void activeContour(Mat magnitude, Mat direction, vector<ContourPoint> &contour, float alpha, float beta, float gamma);

/**
 * @brief Calculates the total energy of a contour.
 *
 * @param contour Vector of ContourPoints.
 * @return Total energy of the contour.
 */
float contourEnergy(vector<ContourPoint> &contour);

/**
 * @brief Visualises the snake on an image.
 *
 * @param image The source image.
 * @param contour Vector of ContourPoints representing the snake.
 * @return Image with the snake visualised.
 */
Mat showSnake(Mat image, vector<ContourPoint> contour);

/**
 * @brief Displays gradient information on an image.
 *
 * @param image Reference to the source image.
 * @param magnitude Gradient magnitude.
 * @param direction Gradient direction.
 * @param stride Stride for visualization.
 */
void showGradient(Mat &image, Mat magnitude, Mat direction, int stride);

#endif // _FUNC_H_
