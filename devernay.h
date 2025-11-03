#pragma once
#include <opencv2/opencv.hpp>


void devernay(double** x, double** y, int* N, int** curve_limits, int* M,
    cv::Mat& image, int X, int Y,
    double sigma, double th_h, double th_l); 
