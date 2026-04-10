cmake_minimum_required(VERSION 3.8)
project(ShowUndistortedProject)

# OpenCV 패키지 찾기
find_package(OpenCV REQUIRED)

# 실행 파일 대상 추가
add_executable(ShowUndistorted ShowUndistorted.cpp)

# OpenCV 라이브러리를 실행 파일에 링크
target_link_libraries(ShowUndistorted ${OpenCV_LIBS})