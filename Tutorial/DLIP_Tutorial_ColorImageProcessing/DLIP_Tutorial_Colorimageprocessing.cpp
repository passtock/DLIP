//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <ctype.h>
#include <iostream>
#include <deque>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat image;
Point origin;
Rect selection;
bool selectObject = false;
bool trackObject = false;
int hmin = 1, hmax = 179, smin = 30, smax = 255, vmin = 0, vmax = 255;

/// On mouse event 
static void onMouse(int event, int x, int y, int, void*);

// 사각형 영역과 기록된 시간을 저장하기 위한 구조체
struct TrackPoint {
	Rect box;
	double timestamp;
};

int main()
{
	Mat image_disp, hsv, hue, mask, dst;
	vector<vector<Point> > contours;

	// 웹캠 열기
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "웹캠을 열 수 없습니다." << endl;
		return -1;
	}

	cap >> image; // 초기 프레임을 읽어와서 사이즈 설정을 위해 사용
	Mat dst_track = Mat::zeros(image.size(), CV_8UC3);

	// 최근 10초간의 경로와 시간을 저장할 덱(Deque)
	deque<TrackPoint> trackPoints;

	// TrackBar 설정
	namedWindow("Source", 0);
	setMouseCallback("Source", onMouse, 0);
	createTrackbar("Hmin", "Source", &hmin, 179, 0);
	createTrackbar("Hmax", "Source", &hmax, 179, 0);
	createTrackbar("Smin", "Source", &smin, 255, 0);
	createTrackbar("Smax", "Source", &smax, 255, 0);
	createTrackbar("Vmin", "Source", &vmin, 255, 0);
	createTrackbar("Vmax", "Source", &vmax, 255, 0);

	while (true)
	{
		cap >> image; // 실시간 영상 프레임 가져오기
		if (image.empty()) break;

		// 사용자가 거울처럼 볼 수 있도록 좌우 반전 (선택 사항)
		flip(image, image, 1);
		image.copyTo(image_disp);

		imshow("Source", image);

		/******** Convert BGR to HSV ********/
		cvtColor(image, hsv, COLOR_BGR2HSV);

		/******** Add Pre-Processing such as filtering etc  ********/
		GaussianBlur(hsv, hsv, Size(5, 5), 0);

		/// set dst as the output of InRange
		inRange(hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
			Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), dst);

		/******** Add Post-Processing such as morphology etc  ********/
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		morphologyEx(dst, dst, MORPH_OPEN, kernel);
		morphologyEx(dst, dst, MORPH_CLOSE, kernel);

		namedWindow("InRange", 0);
		imshow("InRange", dst);

		/// once mouse has selected an area bigger than 0
		if (trackObject)
		{
			trackObject = false;					// Terminate the next Analysis loop
			Mat roi_HSV(hsv, selection); 			// Set ROI by the selection box		
			Scalar means, stddev;
			meanStdDev(roi_HSV, means, stddev);
			cout << "\n Selected ROI Means= " << means << " \n stddev= " << stddev;

			// Change the value in the trackbar according to Mean and STD //
			hmin = MAX((means[0] - stddev[0] * 2), 0);
			hmax = MIN((means[0] + stddev[0] * 2), 179);
			setTrackbarPos("Hmin", "Source", hmin);
			setTrackbarPos("Hmax", "Source", hmax);

			/******** Repeat for S and V trackbar ********/
			smin = MAX((means[1] - stddev[1] * 2), 0);
			smax = MIN((means[1] + stddev[1] * 2), 255);
			vmin = MAX((means[2] - stddev[2] * 2), 0);
			vmax = MIN((means[2] + stddev[2] * 2), 255);
			setTrackbarPos("Smin", "Source", smin);
			setTrackbarPos("Smax", "Source", smax);
			setTrackbarPos("Vmin", "Source", vmin);
			setTrackbarPos("Vmax", "Source", vmax);

			// 새로운 객체를 선택할 때마다 기존 잔상 초기화
			trackPoints.clear();
		}

		if (selectObject && selection.area() > 0)  // Left Mouse is being clicked and dragged
		{
			// Mouse Drag을 화면에 보여주기 위함
			Mat roi_RGB(image_disp, selection);
			bitwise_not(roi_RGB, roi_RGB);
			imshow("Source", image_disp);
		}
		image.copyTo(image_disp);

		///  Find All Contour   ///
		findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		// 현재 시간 측정 (초 단위)
		double currentTime = (double)getTickCount() / getTickFrequency();

		if (contours.size() > 0)
		{
			/// Find the Contour with the largest area ///
			double maxArea = 0;
			int maxArea_idx = 0;

			for (int i = 0; i < contours.size(); i++)
				if (contourArea(contours[i]) > maxArea) {
					maxArea = contourArea(contours[i]);
					maxArea_idx = i;
				}

			// 일정 크기 이상일 때만 추적 (노이즈 방지)
			if (maxArea > 50)
			{
				Rect boxPoint = boundingRect(contours[maxArea_idx]);

				// 현재 박스와 시간을 덱에 저장
				trackPoints.push_back({ boxPoint, currentTime });

				/// Draw the Contour Box on Original Image ///
				// 현재 위치는 눈에 띄게 자홍색(Magenta) 굵은 네모로 표시
				rectangle(image_disp, boxPoint, Scalar(255, 0, 255), 3);
			}
		}

		// 시간 초과된(10초 이전) 궤적 포인트 제거
		while (!trackPoints.empty() && (currentTime - trackPoints.front().timestamp) > 10.0) {
			trackPoints.pop_front();
		}

		// 매 프레임마다 새롭게 저장된 궤적(박스) 그리기
		Mat trajectory = Mat::zeros(image.size(), CV_8UC3);
		for (size_t i = 0; i < trackPoints.size(); i++) {
			// 과거의 위치 잔상은 초록색(Green)의 얇은 네모로 표시
			rectangle(trajectory, trackPoints[i].box, Scalar(255, 0, 255), 3);
		}

		// 원본 이미지(또는 현재 박스 그려진 이미지)에 10초 캔버스를 더함
		image_disp = image_disp + trajectory;

		namedWindow("Contour_Box", 0);
		imshow("Contour_Box", image_disp);

		char c = (char)waitKey(10);
		if (c == 27)
			break;
		else if (c == 'c' || c == 'C') // 'c' 키를 누르면 궤적 지우기 기능
		{
			trackPoints.clear();
		}
	} // end of for(;;)

	return 0;
}


/// On mouse event 
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)  // for any mouse motion
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = abs(x - origin.x) + 1;
		selection.height = abs(y - origin.y) + 1;
		selection &= Rect(0, 0, image.cols, image.rows);  /// Bitwise AND  check selectin is within the image coordinate
	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		selectObject = true;
		origin = Point(x, y);
		break;
	case EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.area())
			trackObject = true;
		break;
	}
}