

// 使用Opencv 重写Canny 亚像素边缘提取


#include <iostream>
#include <opencv2/opencv.hpp>
#include "devernay.h"



int main() {

	// std::string path = "E:\\Code\\C++_Code\\Canny-Devernay\\data\\image.pgm";        // 图像路径
	std::string path = "E:\\Code\\C++_Code\\Canny-Devernay\\data\\circle.png";
	cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE); // 以灰度图的方式读取数据

	int X = image.cols;    // 宽
	int Y = image.rows;    // 高

	cv::Mat image_d64;
	image.convertTo(image_d64, CV_64F);  // 类型转换

	double* x, * y;
	int N, M;
	int* curve_limits;
	double S = 2.0, H = 10.0, L = 5.0;
	devernay(&x, &y, &N, &curve_limits, &M, image_d64, X, Y, S, H, L);



	// 创建彩色显示图
	//cv::Mat display;
	//cv::cvtColor(cv::Mat::zeros(Y,X,CV_8UC1), display, cv::COLOR_GRAY2BGR);

	// 绘制曲线
	//for (int m = 0; m < M; ++m) {
	//	int start = curve_limits[m];
	//	int end = curve_limits[m + 1];
	//	for (int i = start; i < end - 1; ++i) {
	//		cv::Point2d p1(x[i], y[i]);
	//		cv::Point2d p2(x[i + 1], y[i + 1]);
	//		cv::line(display, p1, p2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
	//	}
	//}

	// 绘制亚像素点（可选）
	//for (int i = 0; i < N; ++i) {
	//	cv::circle(display, cv::Point2d(x[i], y[i]), 0.5, cv::Scalar(0, 0, 255), -1);
	//}

	//cv::imshow("Devernay Subpixel Edges", display);
	//cv::imwrite("Devernay.bmp", display);
	//cv::waitKey(0);

}