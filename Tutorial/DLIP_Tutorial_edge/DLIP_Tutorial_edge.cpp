#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 원본/전처리 이미지
Mat src1, src2;
Mat gray1, gray2;

// 결과 출력용
Mat display1, display2;

// 창 이름
const string WIN_TRAFFIC = "TrafficSign";
const string WIN_EYE = "EyePupil";

// TrafficSign용 HoughCircles 파라미터
int dp1_x10 = 10;      // 실제 dp = dp1_x10 / 10.0
int minDist1 = 50;
int param1_1 = 150;
int param2_1 = 40;
int minRadius1 = 50;
int maxRadius1 = 200;

// EyePupil용 HoughCircles 파라미터 (홍채 탐지용으로 사용)
int dp2_x10 = 20;      // 초기값 세팅
int minDist2 = 380;
int param1_2 = 106;
int param2_2 = 4;
int minRadius2 = 30;
int maxRadius2 = 100;

void detectAndShowTraffic(int, void*)
{
    if (src1.empty() || gray1.empty()) return;

    Mat blurred;
    GaussianBlur(gray1, blurred, Size(9, 9), 2, 2);

    vector<Vec3f> circles1;

    double dp = max(dp1_x10, 1) / 10.0;
    int minD = max(minDist1, 1);
    int p1 = max(param1_1, 1);
    int p2 = max(param2_1, 1);
    int minR = min(minRadius1, maxRadius1);
    int maxR = max(minRadius1, maxRadius1);

    HoughCircles(
        blurred,
        circles1,
        HOUGH_GRADIENT,
        dp, minD, p1, p2, minR, maxR
    );

    display1 = src1.clone();

    for (size_t i = 0; i < circles1.size(); i++)
    {
        Point center(cvRound(circles1[i][0]), cvRound(circles1[i][1]));
        int radius = cvRound(circles1[i][2]);

        // 중심점은 초록색
        circle(display1, center, 3, Scalar(0, 255, 0), -1, LINE_AA);
        // 표지판 테두리는 눈에 잘 띄게 노란색(0, 255, 255), 두께 4로 변경
        circle(display1, center, radius, Scalar(0, 255, 255), 4, LINE_AA);
    }

    putText(display1,
        format("dp=%.1f minDist=%d p1=%d p2=%d minR=%d maxR=%d",
            dp, minD, p1, p2, minR, maxR),
        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

    putText(display1,
        format("Traffic Signs = %d", (int)circles1.size()),
        Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

    imshow(WIN_TRAFFIC, display1);
}

void detectAndShowEye(int, void*)
{
    if (src2.empty() || gray2.empty()) return;

    Mat blurred;
    GaussianBlur(gray2, blurred, Size(9, 9), 2, 2);

    display2 = src2.clone();

    // ---------------------------------------------------------
    // 1. 동공 (Pupil) 탐지 - 캡처화면에서 찾은 파라미터 고정 적용
    // ---------------------------------------------------------
    vector<Vec3f> pupils;
    HoughCircles(
        blurred, pupils, HOUGH_GRADIENT,
        2.0,    // dp
        380,    // minDist
        106,    // param1
        4,      // param2
        4,      // minRadius (작은 원)
        25      // maxRadius
    );

    for (size_t i = 0; i < pupils.size(); i++)
    {
        Point center(cvRound(pupils[i][0]), cvRound(pupils[i][1]));
        int radius = cvRound(pupils[i][2]);
        // 동공은 빨간색 테두리로 그리기
        circle(display2, center, 2, Scalar(0, 255, 0), -1, LINE_AA);
        circle(display2, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
    }

    // ---------------------------------------------------------
    // 2. 홍채 (Iris) 탐지 - 트랙바 파라미터와 연동
    // ---------------------------------------------------------
    vector<Vec3f> irises;
    double dp = max(dp2_x10, 1) / 10.0;
    int minD = max(minDist2, 1);
    int p1 = max(param1_2, 1);
    int p2 = max(param2_2, 1);
    int minR = min(minRadius2, maxRadius2);
    int maxR = max(minRadius2, maxRadius2);

    HoughCircles(
        blurred, irises, HOUGH_GRADIENT,
        dp, minD, p1, p2, minR, maxR
    );

    for (size_t i = 0; i < irises.size(); i++)
    {
        Point center(cvRound(irises[i][0]), cvRound(irises[i][1]));
        int radius = cvRound(irises[i][2]);
        // 홍채는 파란색 테두리로 그리기
        circle(display2, center, 2, Scalar(0, 255, 255), -1, LINE_AA);
        circle(display2, center, radius, Scalar(255, 0, 0), 3, LINE_AA);
    }

    putText(display2,
        format("Iris Trackbar: dp=%.1f minDist=%d p1=%d p2=%d minR=%d maxR=%d",
            dp, minD, p1, p2, minR, maxR),
        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

    putText(display2,
        format("Pupils (Red) = %d, Irises (Blue) = %d", (int)pupils.size(), (int)irises.size()),
        Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

    imshow(WIN_EYE, display2);
}

int main(int argc, char** argv)
{
    String filename1 = "../../image/TrafficSign1.png";
    String filename2 = "../../image/eyepupil.png";

    src1 = imread(filename1, IMREAD_COLOR);
    src2 = imread(filename2, IMREAD_COLOR);

    if (src1.empty())
    {
        printf("Error opening TrafficSign image\n");
        return -1;
    }

    if (src2.empty())
    {
        printf("Error opening EyePupil image\n");
        return -1;
    }

    cvtColor(src1, gray1, COLOR_BGR2GRAY);
    cvtColor(src2, gray2, COLOR_BGR2GRAY);

    namedWindow(WIN_TRAFFIC, WINDOW_AUTOSIZE);
    namedWindow(WIN_EYE, WINDOW_AUTOSIZE);

    // TrafficSign 창 트랙바
    createTrackbar("dp x10", WIN_TRAFFIC, &dp1_x10, 50, detectAndShowTraffic);
    createTrackbar("minDist", WIN_TRAFFIC, &minDist1, 500, detectAndShowTraffic);
    createTrackbar("param1", WIN_TRAFFIC, &param1_1, 500, detectAndShowTraffic);
    createTrackbar("param2", WIN_TRAFFIC, &param2_1, 500, detectAndShowTraffic);
    createTrackbar("minRadius", WIN_TRAFFIC, &minRadius1, 500, detectAndShowTraffic);
    createTrackbar("maxRadius", WIN_TRAFFIC, &maxRadius1, 500, detectAndShowTraffic);

    // EyePupil 창 트랙바 (홍채 탐지용)
    createTrackbar("dp x10", WIN_EYE, &dp2_x10, 50, detectAndShowEye);
    createTrackbar("minDist", WIN_EYE, &minDist2, 500, detectAndShowEye);
    createTrackbar("param1", WIN_EYE, &param1_2, 500, detectAndShowEye);
    createTrackbar("param2", WIN_EYE, &param2_2, 500, detectAndShowEye);
    createTrackbar("minRadius", WIN_EYE, &minRadius2, 500, detectAndShowEye);
    createTrackbar("maxRadius", WIN_EYE, &maxRadius2, 500, detectAndShowEye);

    // 초기 결과 표시
    detectAndShowTraffic(0, 0);
    detectAndShowEye(0, 0);

    while (true)
    {
        int key = waitKey(30);
        if (key == 27) break; // ESC 종료
    }

    destroyAllWindows();
    return 0;
}