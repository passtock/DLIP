#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int hmin = 3, hmax = 31;
int smin = 70, smax = 190;
int vmin = 110, vmax = 250;

int current_frame = 0;
int total_frames = 0;

int main()
{
	Mat image, image_disp, hsv, mask1, mask2, res1, res2, final_output;
	Mat background;

	VideoCapture cap("LAB_MagicCloak_Sample1.mp4");
	if (!cap.isOpened()) {
		cout << "동영상을 열 수 없습니다." << endl;
		return -1;
	}

	total_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);

	// =================================================================
	// 💡 [해결책 1] 깨끗한 배경 캡처 및 디코더(잔상) 꼬임 방지
	// =================================================================
	// 1. 사람이 완전히 빠져나간 31번째 프레임으로 이동
	cap.set(CAP_PROP_POS_FRAMES, 31);

	// 2. 31번 프레임을 배경으로 캡처 후 안전하게 복사
	cap >> background;
	if (background.empty()) {
		cout << "배경 이미지를 불러올 수 없습니다." << endl;
		return -1;
	}
	background = background.clone();

	// 3. 비디오 디코더가 꼬이지 않도록 객체를 완전히 닫았다가 다시 열기 (0프레임부터 시작)
	cap.release();
	cap.open("LAB_MagicCloak_Sample1.mp4");
	// =================================================================

	namedWindow("Source", WINDOW_AUTOSIZE);
	createTrackbar("Frame", "Source", &current_frame, total_frames, 0);
	createTrackbar("Hmin", "Source", &hmin, 179, 0);
	createTrackbar("Hmax", "Source", &hmax, 179, 0);
	createTrackbar("Smin", "Source", &smin, 255, 0);
	createTrackbar("Smax", "Source", &smax, 255, 0);
	createTrackbar("Vmin", "Source", &vmin, 255, 0);
	createTrackbar("Vmax", "Source", &vmax, 255, 0);

	namedWindow("Mask", WINDOW_AUTOSIZE);

	while (true)
	{
		cap >> image;
		if (image.empty()) break;

		current_frame = (int)cap.get(CAP_PROP_POS_FRAMES);
		setTrackbarPos("Frame", "Source", current_frame);

		image.copyTo(image_disp);
		imshow("Source", image_disp);

		// HSV 변환 및 마스크 생성
		cvtColor(image, hsv, COLOR_BGR2HSV);
		GaussianBlur(hsv, hsv, Size(5, 5), 0);

		inRange(hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
			Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), mask1);

		// 노이즈 제거 및 경계선 다듬기
		Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
		morphologyEx(mask1, mask1, MORPH_OPEN, kernel);
		morphologyEx(mask1, mask1, MORPH_DILATE, kernel, Point(-1, -1), 5);

		imshow("Mask", mask1);

		bitwise_not(mask1, mask2);

		// 합성 전, 과거의 잔상이 남지 않도록 도화지(res1, res2)를 초기화!
		res1 = Mat::zeros(image.size(), image.type());
		res2 = Mat::zeros(image.size(), image.type());

		// 1. 타겟 영역(mask1)에는 '31프레임 배경' 덮어씌우기
		bitwise_and(background, background, res1, mask1);

		// 2. 나머지 영역(mask2)은 '현재 프레임' 원본 유지하기
		bitwise_and(image, image, res2, mask2);

		// 3. 최종 합성
		add(res1, res2, final_output);

		namedWindow("Magic Cloak", WINDOW_AUTOSIZE);
		imshow("Magic Cloak", final_output);

		char c = (char)waitKey(30);
		if (c == 27) break;
	}

	cap.release();
	destroyAllWindows();
	return 0;
}