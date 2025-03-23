#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main() {
    // Check if CUDA is available
    if(!cv::cuda::getCudaEnabledDeviceCount()) {
        std::cerr << "CUDA is not available on this device" << std::endl;
        return -1;
    }

    // Load an image
    cv::Mat src = cv::imread("../ASSETS/example2.jpg"); // Replace with an actual image file path
    if (src.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Upload image to GPU
    cv::cuda::GpuMat d_src, d_dst;
    d_src.upload(src);

    // Convert to grayscale using GPU
    cv::cuda::cvtColor(d_src, d_dst, cv::COLOR_BGR2GRAY);

    // Download result back to CPU
    cv::Mat dst;
    d_dst.download(dst);

    // Save the result
    cv::imwrite("../ASSETS/gray_image_gpu.jpg", dst);
    std::cout << "Converted image to grayscale on GPU and saved as 'gray_image_gpu.jpg'" << std::endl;

    return 0;
}
