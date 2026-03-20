#include <opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//* @function main
int main()
{
	Mat src;

	src = imread("testImage.jpg", 1);/// Load an image

	if (src.empty())/// Load image check
	{
		cout << "File Read Failed : src is empty" << endl;
		waitKey(0);
	}

	/// Create a window to display results
	namedWindow("DemoWIndow", WINDOW_AUTOSIZE); //WINDOW_AUTOSIZE(1) :Fixed Window, 0: Unfixed window

	if (!src.empty())imshow("DemoWIndow", src); // Show image

	waitKey(0);//Pause the program
	return 0;
}