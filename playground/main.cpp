// file: main.cpp

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

// Prototype for kernel (could also be in a header file)
__global__ void invertColorsKernel(uchar3 *input, uchar3 *output, int width, int height);

int main() {
    cv::Mat src = cv::imread("path_to_image.jpg");
    if (src.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    cv::cuda::GpuMat d_src, d_dst;
    d_src.upload(src);

    // Convert to uchar3 for kernel operation
    cv::cuda::GpuMat d_src_uchar3, d_dst_uchar3;
    d_src.convertTo(d_src_uchar3, CV_8UC3);
    d_dst_uchar3.create(d_src.size(), CV_8UC3);

    // Launch the kernel
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
    invertColorsKernel<<<grid, block>>>(d_src_uchar3.ptr<uchar3>(), d_dst_uchar3.ptr<uchar3>(), src.cols, src.rows);

    // Convert back to the original type and download to host
    d_dst_uchar3.convertTo(d_dst, d_src.type());
    cv::Mat dst;
    d_dst.download(dst);

    // Display the result
    cv::imshow("Original", src);
    cv::imshow("Inverted", dst);
    cv::waitKey(0);

    return 0;
}
