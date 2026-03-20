#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	/*  read image  */
	Mat img = imread("../../Image/image.jpg");
	int W = img.cols;
	int H = img.rows;
	if (img.empty())/// Load image check
	{
		cout << "File Read Failed : src is empty" << endl;
		waitKey(0);
	}

	imshow("img", img);

	/*  Crop(Region of Interest)  */
	Rect r(0, 0, img.cols, img.rows);	 // (x, y, width, height)
	Rect bound(130, 140, 190, 200);	// (x, y, width, height)
	Mat roiImg = img(r&bound);
	Mat background = Mat(H, W, CV_8UC3, Scalar(0));
	roiImg.copyTo(background(r & bound));
	
	imshow("roiImg", background);

	/*  Rotate  */
	Mat rotImg;
	rotate(img, rotImg, ROTATE_90_CLOCKWISE);
	imshow("rotImg", rotImg);

	/*  Resize  */
	Mat resizedImg;
	resize(img, resizedImg, Size(img.cols / 2, img.rows / 2));
	imshow("resizedImg", resizedImg);

	waitKey(0);
}