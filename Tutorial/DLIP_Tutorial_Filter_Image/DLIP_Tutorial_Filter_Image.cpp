#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	String HGU_logo = "C:\\Users\\passp\\Desktop\\coding\\psa grading\\data\\raw\\PSA8\\psa8_33c420901b.jpg";
	Mat src = imread(HGU_logo);

	if (src.empty())/// Load image check
	{
		cout << "File Read Failed : src is empty" << endl;
		waitKey(0);
	}

	Mat src_gray = imread(HGU_logo, 0);  // read in grayscale
	

	/*  write image  */
	String fileName = "writeImage.jpg";
	imwrite(fileName, src);

	/*  display image  */
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);

	

	/*int i = 3;
	Mat blurcircuit = src.clone();
	blur(src, blurcircuit, Size(i, i), Point(-1, -1));
	imshow("blurcircuit", blurcircuit);

	Mat dst;
	int ddepth = -1;
	Mat kernel = (Mat_<float>(5, 5) << 0, 0, -1, 0, 0,
		0, 1, 3, 1, 0,
		0, 3, 5, 3, 0,
		0, 1, 3, 1, 0,
		0, 0, -1, 0,  0);
	Point anchor = Point(-1, -1);
	filter2D(src, dst, ddepth, kernel, anchor);
	imshow("dst", dst);

	Mat dst1;
	GaussianBlur(src, dst1, Size(3, 3), 0, 0);
	imshow("dst1", dst1);

	Mat dst2;
	medianBlur(src, dst2, 3);
	imshow("dst2", dst2);

	Mat dst3;
	bilateralFilter(src, dst3, 9, 75, 75);
	imshow("dst3", dst3);*/

	int kernel_size1 = 3;
	int scale1 = 1;
	int delta1 = 0;
	int ddepth1 = CV_16S;
	Mat dst4, result_laplcaian;
	Laplacian(src, dst4, ddepth1, kernel_size1, scale1, delta1, BORDER_DEFAULT);
	src.convertTo(src, CV_16S);
	result_laplcaian = src + dst4;
	result_laplcaian.convertTo(result_laplcaian, CV_8U);
	imshow("result_laplcaian", result_laplcaian);

	int kernel_size2 = 3;
	int delta2 = 3;
	Mat dst5, result_laplcaian1;
	Laplacian(src, dst5, ddepth1, kernel_size2, scale1, delta2, BORDER_DEFAULT);
	src.convertTo(src, CV_16S);
	result_laplcaian1 = src - dst5;
	result_laplcaian1.convertTo(result_laplcaian1, CV_8U);
	imshow("result_laplcaian1", result_laplcaian1);
	waitKey(0);
}