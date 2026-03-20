#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main() {
    vector<string> file_names = {"Gear1.jpg", "Gear2.jpg", "Gear3.jpg", "Gear4.jpg"};

    for (int idx = 0; idx < file_names.size(); idx++) {
        string file_name = file_names[idx];
        string prefix = "[Gear" + to_string(idx + 1) + "] "; 

        // 1. 이미지 읽기
        Mat img_color = imread(file_name, IMREAD_COLOR);
        if (img_color.empty()) {
            cerr << prefix << "이미지를 불러올 수 없습니다." << endl;
            continue; // 다음 이미지로 넘어감
        }

        Mat img_gray;
        cvtColor(img_color, img_gray, COLOR_BGR2GRAY);

        // ---------------------------------------------------------
        // 2. Laplacian 필터 및 이진화, 팽창
        // ---------------------------------------------------------
        Mat img_laplacian, edges_binary;
        Laplacian(img_gray, img_laplacian, CV_16S, 3);
        convertScaleAbs(img_laplacian, img_laplacian);
        threshold(img_laplacian, edges_binary, 30, 255, THRESH_BINARY);

        Mat dilate_kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        dilate(edges_binary, edges_binary, dilate_kernel);

        // ---------------------------------------------------------
        // 3. 솔리드(Solid) 마스크 생성
        // ---------------------------------------------------------
        vector<vector<Point>> edge_contours;
        findContours(edges_binary, edge_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat mask_solid = Mat::zeros(img_gray.size(), CV_8U);
        drawContours(mask_solid, edge_contours, -1, Scalar(255), FILLED);

        // ---------------------------------------------------------
        // 4. 거리 변환으로 뿌리원 몸체 분리
        // ---------------------------------------------------------
        Mat dist;
        distanceTransform(mask_solid, dist, DIST_L2, 5);

        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(dist, &minVal, &maxVal, &minLoc, &maxLoc);

        Mat mask_body = Mat::zeros(img_gray.size(), CV_8U);
        int root_radius = cvRound(maxVal) + 1;
        circle(mask_body, maxLoc, root_radius, Scalar(255), FILLED);

        Mat mask_teeth;
        subtract(mask_solid, mask_body, mask_teeth);

        Mat small_kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(mask_teeth, mask_teeth, MORPH_OPEN, small_kernel);

        // ---------------------------------------------------------
        // 5. 이빨 면적 계산 및 불량 판정
        // ---------------------------------------------------------
        vector<vector<Point>> all_teeth_contours;
        findContours(mask_teeth, all_teeth_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<vector<Point>> contours;
        for (size_t i = 0; i < all_teeth_contours.size(); i++) {
            if (contourArea(all_teeth_contours[i]) > 15.0) {
                contours.push_back(all_teeth_contours[i]);
            }
        }

        double total_area = 0.0;
        for (size_t i = 0; i < contours.size(); i++) {
            total_area += contourArea(contours[i]);
        }
        double avg_area = contours.empty() ? 0.0 : total_area / contours.size();
        
        // 정상 이빨 기준 (평균의 90% ~ 110%)
        double min_normal_area = avg_area * 0.9; 
        double max_normal_area = avg_area * 1.1; 

        // ---------------------------------------------------------
        // 6. 결과 렌더링용 캔버스 생성
        // ---------------------------------------------------------
        // ★ 수정됨: 2, 3번 이미지는 원본을 복사하지 않고 완전히 까만 배경(Zeros) 사용
        Mat img_res2 = Mat::zeros(img_color.size(), CV_8UC3); // 까만 배경 + 이빨선 + 텍스트
        Mat img_res3 = Mat::zeros(img_color.size(), CV_8UC3); // 까만 배경 + 이빨선 (텍스트 없음)
        Mat img_res4 = img_color.clone(); // 4번째는 원본 배경 위에 불량 표기

        int defect_count = 0;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            
            bool is_defect = (area < min_normal_area || max_normal_area < area);
            
            if (is_defect) defect_count++;

            Scalar color = is_defect ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

            drawContours(img_res2, contours, (int)i, color, 2);
            drawContours(img_res3, contours, (int)i, color, 2);
            
            if (is_defect) {
                drawContours(img_res4, contours, (int)i, Scalar(0, 0, 255), 2);
            }

            Moments M = moments(contours[i]);
            if (M.m00 != 0) {
                int cX = int(M.m10 / M.m00);
                int cY = int(M.m01 / M.m00);
                putText(img_res2, to_string((int)area), Point(cX - 15, cY + 5), FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            }
        }

        cout << "\n\n[" << file_name << "] 분석 결과" << endl;
        cout << "============================================" << endl;
        cout << "| " << left << setw(10) << "이빨 개수" 
             << "| " << setw(12) << "평균 면적" 
             << "| " << setw(12) << "불량 개수" << " |" << endl;
        cout << "--------------------------------------------" << endl;
        cout << "| " << left << setw(10) << contours.size() 
             << "| " << setw(12) << fixed << setprecision(1) << avg_area 
             << "| " << setw(12) << defect_count << " |" << endl;
        cout << "============================================" << endl;

        imshow(prefix + "1. Original", img_color);
        imshow(prefix + "2. Teeth Area Only", img_res2);
        imshow(prefix + "3. Teeth Outlines Only", img_res3);
        imshow(prefix + "4. Missing Teeth", img_res4);
    }

    cout << "\n>> 모든 분석이 완료되었습니다. 창에서 아무 키나 누르면 종료됩니다..." << endl;
    waitKey(0);

    return 0;
}