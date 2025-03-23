// test_opencv.cpp
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main() {
    // Load an image
    cv::Mat colorImage = cv::imread("../ASSETS/example.jpg"); // Replace with an actual image file path
    if(colorImage.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Convert to grayscale
    cv::Mat grayImage;
    cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);

    // Save the result
    cv::imwrite("../ASSETS/gray_image.jpg", grayImage);
    std::cout << "Converted image to grayscale and saved as 'gray_image.jpg'" << std::endl;

    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    return 0;
}
