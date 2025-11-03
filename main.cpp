

// 使用Opencv 重写Canny 亚像素边缘提取


#include <iostream>
#include <opencv2/opencv.hpp>
#include "devernay.h"



int main() {

	std::string path = "E:\\Code\\C++_Code\\Canny-Devernay\\data\\image.pgm";        // 图像路径

	cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE); // 以灰度图的方式读取数据


	int X = image.cols;    // 宽
	int Y = image.rows;    // 高

	cv::Mat image_d64;
	image.convertTo(image_d64, CV_64F);  // 类型转换

	//cv::imshow("df", image_d64);
	//cv::waitKey();


	double* x, * y;
	int N, M;
	int* curve_limits;
	double S = 1.0, H = 10.0, L = 5.0;
	devernay(&x, &y, &N, &curve_limits, &M, image_d64, X, Y, S, H, L);

}