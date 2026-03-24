cmake_minimum_required(VERSION 3.10)
project(DLIP_LAB_GEAR)

set(CMAKE_CXX_STANDARD 14)

# ★ 추가된 부분: 한글 깨짐 및 컴파일 오류(C4819, C2001) 방지를 위해 MSVC에 UTF-8 인코딩 강제
if(MSVC)
    add_compile_options(/utf-8)
endif()

# CMake의 엄격한 컴파일러 버전(VC17) 체크로 인해 OpenCVConfig.cmake에서 발생하는 오류를 회피하기 위해,
# 기존 Visual Studio 설정(.props)에 있던 경로를 수동으로 직접 지정해줍니다.

set(OPENCV_INC "C:/opencv/build/include")
set(OPENCV_LIB "C:/opencv/build/x64/vc16/lib")

# 만약 opencv-4.9.0 폴더에 설치된 환경이라면 경로 우선 적용
if(EXISTS "C:/opencv-4.9.0/build/include")
    set(OPENCV_INC "C:/opencv-4.9.0/build/include")
    set(OPENCV_LIB "C:/opencv-4.9.0/build/x64/vc16/lib")
endif()

include_directories(${OPENCV_INC})
link_directories(${OPENCV_LIB})

add_executable(LAB_GEAR LAB_GEAR.cpp)

# 링커 설정 (디버그 모드일 때는 490d, 릴리즈일 때는 490)
target_link_libraries(LAB_GEAR
    debug opencv_world490d
    optimized opencv_world490
)
