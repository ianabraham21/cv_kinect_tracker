cmake_minimum_required(VERSION 2.8.3)
project(tracker)

add_definitions(-std=c++14)

find_package(OpenCV REQUIRED COMPONENTS core highgui)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(ball_tracker src/main_tracker.cpp)
target_link_libraries(ball_tracker pthread usb-1.0 freenect ${OpenCV_LIBS} opencv_imgproc opencv_imgcodecs)

#include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(calibration src/calibration.cpp)
target_link_libraries(calibration pthread usb-1.0 freenect ${OpenCV_LIBS} opencv_imgproc opencv_imgcodecs)