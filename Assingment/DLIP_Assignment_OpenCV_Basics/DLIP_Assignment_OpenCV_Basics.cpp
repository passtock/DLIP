#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
	Mat img = imread("../../Image/HGU_logo.jpg");
	if (img.empty()) {
		cout << "Error: Image not found!" << endl;
		return -1;
	}
	Mat Out1 = Mat(img.rows, img.cols, CV_8UC1, Scalar(255));
	int width = img.cols;
	int height = img.rows;
	Mat img_gray = Mat(height, width, CV_8UC1);
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	imshow("Original", img_gray);

	Mat resized_img;
	resize(img_gray, resized_img, Size(width / 2, height / 2));
	resized_img.copyTo(Out1(Rect(0, 0, width / 2, height / 2)));
	imshow("Output #1", Out1);

	Mat rotate180, background2;
	rotate(img_gray, rotate180, ROTATE_180);
	resize(rotate180, background2, Size(width / 2, height / 2));	
	Mat Out2 = Mat(img.rows, img.cols, CV_8UC1, Scalar(255));
	background2.copyTo(Out2(Rect(width / 4, height / 4, width / 2, height / 2)));
	imshow("Output #2", Out2);

	Mat background3;
	Rect r(130, 140, 190, 200);
	Mat cropped = img_gray(r);
	resize(cropped, background3, Size(width / 3, height / 3));
	Mat Out3 = Mat(img.rows, img.cols, CV_8UC1, Scalar(255));
	background3.copyTo(Out3(Rect(width - width / 3, height - height / 3, width / 3, height / 3)));
	imshow("Output #3", Out3);
	
	waitKey(0);

}