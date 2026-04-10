#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

// ===================== Function Declarations =====================

// Apply Laplacian edge detection and return binary edge map
Mat detectEdges(const Mat& img_gray);

// Fill external contours to create a solid gear mask
Mat createSolidMask(const Mat& edges_binary);

// Separate gear body from teeth using distance transform
void separateBodyAndTeeth(const Mat& mask_solid, Mat& mask_teeth);

// Filter small-noise contours and collect valid teeth contours/areas
void filterTeethContours(const vector<vector<Point>>& all_contours,
                         vector<vector<Point>>& contours,
                         vector<double>& areas,
                         double min_area);

// Compute trimmed mean (exclude top/bottom N values)
double calcTrimmedAverage(const vector<double>& areas, int trim_count);

// Draw results on canvases and count defects
int drawResults(const vector<vector<Point>>& contours,
                double min_area, double max_area,
                Mat& img_res2, Mat& img_res3, Mat& img_res4);

// Print analysis table to console
void printResult(const string& file_name, int teeth_count,
                 double avg_area, int defect_count);

// =========================== main() =============================

int main() {
    vector<string> file_names = { "../../Image/Gear1.jpg", "../../Image/Gear2.jpg", "../../Image/Gear3.jpg", "../../Image/Gear4.jpg" };

    for (int idx = 0; idx < (int)file_names.size(); idx++) {
        string file_name = file_names[idx];
        string prefix = "[Gear" + to_string(idx + 1) + "] ";

        // 1. Load image
        Mat img_color = imread(file_name, IMREAD_COLOR);
        if (img_color.empty()) {
            cerr << prefix << "Could not load image." << endl;
            continue;
        }

        Mat img_gray;
        cvtColor(img_color, img_gray, COLOR_BGR2GRAY);

        // 2. Edge detection
        Mat edges_binary = detectEdges(img_gray);

        // 3. Create solid gear mask
        Mat mask_solid = createSolidMask(edges_binary);

        // 4. Separate body and teeth
        Mat mask_teeth;
        separateBodyAndTeeth(mask_solid, mask_teeth);

        // 5. Extract valid teeth contours
        vector<vector<Point>> all_teeth_contours;
        findContours(mask_teeth, all_teeth_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<vector<Point>> contours;
        vector<double> areas;
        filterTeethContours(all_teeth_contours, contours, areas, 15.0);

        // 6. Compute trimmed average and defect thresholds
        double avg_area = calcTrimmedAverage(areas, 3);
        double min_normal_area = avg_area * 0.9;
        double max_normal_area = avg_area * 1.1;

        // 7. Render results
        Mat img_res2 = Mat::zeros(img_color.size(), CV_8UC3);
        Mat img_res3 = Mat::zeros(img_color.size(), CV_8UC3);
        Mat img_res4 = img_color.clone();

        int defect_count = drawResults(contours, min_normal_area, max_normal_area,
                                       img_res2, img_res3, img_res4);

        // 8. Print analysis table
        printResult(file_name, (int)contours.size(), avg_area, defect_count);

        // 9. Display windows
        imshow(prefix + "1. Original", img_color);
        imshow(prefix + "2. Teeth Area Only", img_res2);
        imshow(prefix + "3. Teeth Outlines Only", img_res3);
        imshow(prefix + "4. Missing Teeth", img_res4);
    }

    cout << "\n>> All analyses completed. Press any key to exit..." << endl;
    waitKey(0);

    return 0;
}

// ==================== Function Definitions =======================

Mat detectEdges(const Mat& img_gray) {
    Mat img_laplacian, edges_binary;
    Laplacian(img_gray, img_laplacian, CV_16S, 3);
    convertScaleAbs(img_laplacian, img_laplacian);
    threshold(img_laplacian, edges_binary, 30, 255, THRESH_BINARY);
    return edges_binary;
}

Mat createSolidMask(const Mat& edges_binary) {
    vector<vector<Point>> edge_contours;
    findContours(edges_binary, edge_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat mask_solid = Mat::zeros(edges_binary.size(), CV_8U);
    drawContours(mask_solid, edge_contours, -1, Scalar(255), FILLED);
    return mask_solid;
}

void separateBodyAndTeeth(const Mat& mask_solid, Mat& mask_teeth) {
    // Distance transform to find gear center and root radius
    Mat dist;
    distanceTransform(mask_solid, dist, DIST_L2, 5);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(dist, &minVal, &maxVal, &minLoc, &maxLoc);

    // Create circular body mask
    Mat mask_body = Mat::zeros(mask_solid.size(), CV_8U);
    int root_radius = cvRound(maxVal) + 1;
    circle(mask_body, maxLoc, root_radius, Scalar(255), FILLED);

    // Subtract body from solid to get teeth region
    subtract(mask_solid, mask_body, mask_teeth);

    // Remove small noise with morphological opening
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask_teeth, mask_teeth, MORPH_OPEN, kernel);
}

void filterTeethContours(const vector<vector<Point>>& all_contours,
                         vector<vector<Point>>& contours,
                         vector<double>& areas,
                         double min_area) {
    for (size_t i = 0; i < all_contours.size(); i++) {
        double area = contourArea(all_contours[i]);
        if (area > min_area) {
            contours.push_back(all_contours[i]);
            areas.push_back(area);
        }
    }
}

double calcTrimmedAverage(const vector<double>& areas, int trim_count) {
    if (areas.empty()) return 0.0;

    vector<double> sorted_areas = areas;
    sort(sorted_areas.begin(), sorted_areas.end());

    // Skip trimming if not enough samples
    if ((int)sorted_areas.size() <= trim_count * 2)
        trim_count = 0;

    int valid_count = (int)sorted_areas.size() - (trim_count * 2);
    double sum = 0.0;

    for (int i = trim_count; i < (int)sorted_areas.size() - trim_count; i++)
        sum += sorted_areas[i];

    return valid_count > 0 ? sum / valid_count : 0.0;
}

int drawResults(const vector<vector<Point>>& contours,
                double min_area, double max_area,
                Mat& img_res2, Mat& img_res3, Mat& img_res4) {
    int defect_count = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        bool is_defect = (area < min_area || area > max_area);

        if (is_defect) defect_count++;

        Scalar color = is_defect ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        // Draw on labeled canvas and outline-only canvas
        drawContours(img_res2, contours, (int)i, color, 2);
        drawContours(img_res3, contours, (int)i, color, 2);

        // Mark defects on original image
        if (is_defect)
            drawContours(img_res4, contours, (int)i, Scalar(0, 0, 255), 2);

        // Label area value at contour center
        Moments M = moments(contours[i]);
        if (M.m00 != 0) {
            int cX = (int)(M.m10 / M.m00);
            int cY = (int)(M.m01 / M.m00);
            putText(img_res2, to_string((int)area),
                    Point(cX - 15, cY + 5),
                    FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
    }

    return defect_count;
}

void printResult(const string& file_name, int teeth_count,
                 double avg_area, int defect_count) {
    cout << "\n\n[" << file_name << "] Analysis Result" << endl;
    cout << "============================================" << endl;
    cout << "| " << left << setw(10) << " Teeth Cnt"
         << "| " << setw(12) << " Trimmed Avg"
         << "| " << setw(12) << "Defects Cnt" << " |" << endl;
    cout << "--------------------------------------------" << endl;
    cout << "| " << left << setw(10) << teeth_count
         << "| " << setw(12) << fixed << setprecision(1) << avg_area
         << "| " << setw(12) << defect_count << " |" << endl;
    cout << "============================================" << endl;
}