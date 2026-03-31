#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

#include "kernel.cuh"

int main()
{
    int numberOfTimes = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cv::Mat img = cv::imread("galaxy_raw.png",  cv::IMREAD_GRAYSCALE);

    if (img.empty())
    {
        std::cerr << "Failed to load image\n";
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    size_t size = width * height * channels;
    
    size_t size_debayer = width * height * 3;
    

    // Allocate original image
    unsigned char* d_img;
    cudaMalloc(&d_img, size);
    cudaMemcpy(d_img, img.data, size, cudaMemcpyHostToDevice);
    
    // Allocate output image
    unsigned char* d_debayer;
    cudaMalloc(&d_debayer, size_debayer);

    // Start measuring time
    cudaEventRecord(start, 0);

    for (int i = 0; i < numberOfTimes; i++) {
        // Call CUDA wrapper
        debayer_cuda(d_img, d_debayer, width, height);

        cudaDeviceSynchronize();
    }

        // Stop measuring time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Wait for the event to be recorded

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel took on average: " << milliseconds/((double)numberOfTimes) << " ms" << std::endl;

    cv::Mat output(height, width, CV_8UC3);

    cudaMemcpy(output.data, d_debayer, size_debayer, cudaMemcpyDeviceToHost);

    cv::namedWindow("Original", cv::WINDOW_KEEPRATIO);
    cv::imshow("Original", img);
    cv::resizeWindow("Original", 960, 600);
    cv::namedWindow("CUDA Debayer", cv::WINDOW_KEEPRATIO);
    cv::imshow("CUDA Debayer", output);
    cv::resizeWindow("CUDA Debayer", 960, 600);
    cv::waitKey(0);

    cudaFree(d_img);
    cudaFree(d_debayer);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
