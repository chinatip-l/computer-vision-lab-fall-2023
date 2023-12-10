// file: invert_kernel.cu

__global__ void invertColorsKernel(uchar3 *input, uchar3 *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        uchar3 pix = input[index];
        output[index] = make_uchar3(255 - pix.x, 255 - pix.y, 255 - pix.z);
    }
}
