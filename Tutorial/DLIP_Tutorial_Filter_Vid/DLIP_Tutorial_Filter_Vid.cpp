/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Filter Demo - Video
* Created: 2021-Spring
------------------------------------------------------*/

#include <opencv2/opencv.hpp>
#include <iostream>


using namespace std;
using namespace cv;

int main()
{
	/*  open the video camera no.0  */
	VideoCapture cap(0);

	if (!cap.isOpened())	// if not success, exit the programm
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", WINDOW_AUTOSIZE);

	int key = 0;
	int kernel_size = 11;
	int filter_type = 0;
	while (1)
	{
		Mat src, dst;

		/*  read a new frame from video  */
		bool bSuccess = cap.read(src);

		if (!bSuccess)	// if not success, break loop
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}


		key = waitKeyEx(30);
		if (key == 27) // wait for 'ESC' press for 30ms. If 'ESC' is pressed, break loop
		{
			cout << "ESC key is pressed by user\n";
			break;
		}
		else if (key == 'b' || key == 'B')
		{
			filter_type = 1;	// blur
		}

		else if (key == 'L' || key == 'l')
		{
			filter_type = 2;	// Laplacian
		}
		else if (key == 'M' || key == 'm')
		{
			filter_type = 3;	// Median
		}
		else if (key == 2490368 || key == 'u') // 2490368: 위쪽 방향키(Up Arrow)
		{
			if (kernel_size < 31)
				kernel_size += 2;
		}
		else if (key == 2621440 || key == 'd') // 2621440: 아래쪽 방향키(Down Arrow)
		{
			if (kernel_size > 1)
				kernel_size -= 2;
		}
		else if(key == 'n' || key == 'N')
		{
			filter_type = 0;	// None
		}
		if (filter_type == 1)
			blur(src, dst, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));
		else if (filter_type == 2)
			Laplacian(src, dst, CV_16S, kernel_size, 1, 0, BORDER_DEFAULT);
		else if (filter_type == 3)
			medianBlur(src, dst, kernel_size);
		else
			src.copyTo(dst);

		imshow("MyVideo", dst);
	}
	return 0;
}