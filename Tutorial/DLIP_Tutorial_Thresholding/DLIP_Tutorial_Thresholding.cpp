/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Threshold using Trackbar Demo
* Created: 2021-Spring
------------------------------------------------------*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Global variables for Threshold
int image_number = 0;
int threshold_value = 0;
int threshold_type = 0;
int morphology_type = 0;

int const max_value = 255;
int const max_mtype = 4;
int const max_type = 6; // (기존 4에서 6으로 변경)
int const max_BINARY_value = 255;

// Global variables for Morphology
int element_shape = MORPH_RECT;		// MORPH_RECT, MORPH_ELIPSE, MORPH_CROSS
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));

Mat src, src_gray, dst, dst_morph;


// Trackbar strings
String window_name = "Threshold & Morphology Demo";
String image_name = "Source Image";
String trackbar_type = "TH Type:";// 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Invert";
String trackbar_value = "TH Value";
String trackbar_morph = "Morph Type"; // 0: None \n 1: erode \n 2: dilate \n 3: close \n 4: open";

// Function headers
void imagenumver_demo(int, void*);
void Threshold_Demo(int, void*);
void Morphology_Demo(int, void*);

int main()
{
	// Load an image
	src = imread("../../Image/LocalThresh1.jpg", IMREAD_COLOR);

	// Convert the image to Gray
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	// Create a window to display the results
	namedWindow(window_name, WINDOW_NORMAL);

	// Create trackbar to choose type of threshold
	createTrackbar(image_name, window_name, &image_number, 8, imagenumver_demo);
	createTrackbar(trackbar_type, window_name, &threshold_type, max_type, Threshold_Demo);
	createTrackbar(trackbar_value, window_name, &threshold_value, max_value, Threshold_Demo);
	createTrackbar(trackbar_morph, window_name, &morphology_type, max_mtype, Morphology_Demo);

	// Call the function to initialize
	Threshold_Demo(0, 0);
	Morphology_Demo(0, 0);

	// Wait until user finishes program
	while (true) {
		int c = waitKey(20);
		if (c == 27)
			break;
	}
}


void imagenumver_demo(int, void*)	// default form of callback function for trackbar
{
	switch (image_number) {
	case 0: src = imread("../../Image/LocalThresh1.jpg", IMREAD_COLOR); break;
	case 1: src = imread("../../Image/LocalThresh2.jpg", IMREAD_COLOR); break;
	case 2: src = imread("../../Image/LocalThresh3.jpg", IMREAD_COLOR); break;
	case 3: src = imread("../../Image/barcode2.jpg", IMREAD_COLOR); break;
	case 4: src = imread("../../Image/rice.png", IMREAD_COLOR); break;
	case 5: src = imread("../../Image/coin.jpg", IMREAD_COLOR); break;
	case 6: src = imread("../../Image/roadshadow.jpg", IMREAD_COLOR); break;
	case 7: src = imread("../../Image/Finger_print_gray.tif", IMREAD_COLOR); break;
	case 8: src = imread("../../Image/Septagon_noisy.tif", IMREAD_COLOR); break;
	}
	
	// 이미지가 변경되면 흑백으로 다시 변환하고 다음 단계(Threshold)를 호출합니다.
	if (!src.empty()) {
		cvtColor(src, src_gray, COLOR_BGR2GRAY);
		Threshold_Demo(0, 0); 
	}
}


void Threshold_Demo(int, void*)	// default form of callback function for trackbar
{
	/*
	* 0: Binary
	* 1: Threshold Inverted
	* 2: Threshold Truncated
	* 3: Threshold to Zero
	* 4: Threshold to Zero Inverted
	* 5: Adaptive Threshold (Mean)
	* 6: Adaptive Threshold (Gaussian)
	*/

	if (threshold_type < 5) {
		// 기존 일반 Threshold
		threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);
	}
	else if (threshold_type == 5) {
		// 적응형 Threshold (평균)
		// 블록 사이즈(11)와 상수(2)는 필요에 따라 변경하거나 별도의 트랙바로 설정 가능합니다.
		adaptiveThreshold(src_gray, dst, max_BINARY_value, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
	}
	else if (threshold_type == 6) {
		// 적응형 Threshold (가우시안)
		adaptiveThreshold(src_gray, dst, max_BINARY_value, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
	}
	
	// 화면에 바로 출력하지 않고 다음 연산(Morphology)을 이어서 호출합니다.
	Morphology_Demo(0, 0);
}

void Morphology_Demo(int, void*)  // default form of callback function for trackbar
{
	/*
	* 0: None
	* 1: Erode
	* 2: Dilate
	* 3: Close
	* 4: Open
	*/

	switch (morphology_type) {
	case 0: dst.copyTo(dst_morph);	break;
	case 1: erode(dst, dst_morph, element); break;
	case 2: dilate(dst, dst_morph, element); break;
	case 3: morphologyEx(dst, dst_morph, MORPH_OPEN, element); break;
	case 4: morphologyEx(dst, dst_morph, MORPH_CLOSE, element); break;
	}
	
	// 모든 처리가 끝난 최종 결과(dst_morph)를 화면에 출력합니다.
	imshow(window_name, dst_morph);
}