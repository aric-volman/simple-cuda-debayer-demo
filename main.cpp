#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

#include "kernel.cuh"

int main()
{
    int numberOfTimes = 10;

    cudaEvent_t start, stop, ee_start, ee_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&ee_start);
    cudaEventCreate(&ee_stop);

    cv::Mat img = cv::imread("galaxy_raw.png",  cv::IMREAD_GRAYSCALE);

    if (img.empty())
    {
        std::cerr << "Failed to load image\n";
        return -1;
    }
    
    /** CPU Test **/
     //cv::Mat color_img_cpu;
     
     double elapsed_cpu_sum = 0;
     
     
     cv::Mat color_img_cpu;
     
     
     
     for (int i = 0; i < numberOfTimes; i++) {
     
     
	    int64 start_cpu = cv::getTickCount();
	    // Perform debayering
	    // Common patterns: BayerBG2BGR, BayerGB2BGR, BayerRG2BGR, BayerGR2BGR
	    cv::cvtColor(img, color_img_cpu, cv::COLOR_BayerRG2BGR);
	    
	    int64 end_cpu = cv::getTickCount();
	    
	    
	    auto elapsed_cpu = end_cpu - start_cpu;
	    
	    
	    
	    
	    elapsed_cpu_sum += (end_cpu - start_cpu) / cv::getTickFrequency();
    
    
    }

    /** End cpu test **/
    
    float totalKernTime = 0;
    float totalEETime = 0;
    

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    size_t size = width * height * channels;
    
    size_t size_debayer = width * height * 3;
    

    // Allocate original image
    unsigned char* d_img;
    cudaMalloc(&d_img, size);
    
    // Allocate output image
    unsigned char* d_debayer;
    cudaMalloc(&d_debayer, size_debayer);
    
    
    cv::Mat output(height, width, CV_8UC3);


    for (int i = 0; i < numberOfTimes; i++) {
    
    
   
    
    
    cudaEventRecord(ee_start, 0);
    cudaMemcpy(d_img, img.data, size, cudaMemcpyHostToDevice);
    	
    // Start measuring kernel time
    cudaEventRecord(start, 0);
    
    // Call CUDA wrapper
    debayer_cuda(d_img, d_debayer, width, height);

    cudaDeviceSynchronize();
        
    // Stop measuring kernel time
    cudaEventRecord(stop, 0);
   
    cudaEventSynchronize(stop); // Wait for the event to be recorded

    cudaMemcpy(output.data, d_debayer, size_debayer, cudaMemcpyDeviceToHost);
    
    // Stop measuring kernel time
    cudaEventRecord(ee_stop, 0);
   
    cudaEventSynchronize(ee_stop); // Wait for the event to be recorded
    
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    totalKernTime += milliseconds;
    
    
    float milTotal = 0;
    
    cudaEventElapsedTime(&milTotal, ee_start, ee_stop);
    
    totalEETime += milTotal;
    

}

    std::cout << "Kernel took on average: " << totalKernTime/((double)numberOfTimes) << " ms" << std::endl;
    
       std::cout << "End to end kernel: " << totalEETime/((double)numberOfTimes) << " ms" << std::endl;
    
    std::cout << "CPU debayer time: " << (elapsed_cpu_sum/((double)numberOfTimes))*1000.0 << " ms" << std::endl;

/*
    cv::namedWindow("Original", cv::WINDOW_KEEPRATIO);
    cv::imshow("Original", img);
    cv::resizeWindow("Original", 960, 600);
    cv::namedWindow("CUDA Debayer", cv::WINDOW_KEEPRATIO);
    cv::imshow("CUDA Debayer", output);
    cv::resizeWindow("CUDA Debayer", 960, 600);
    cv::waitKey(0);*/

    cudaFree(d_img);
    cudaFree(d_debayer);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEventDestroy(ee_start);
    cudaEventDestroy(ee_stop);
    return 0;
}
