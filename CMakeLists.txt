cmake_minimum_required(VERSION 3.5)
project(OpenCV_Camera_Rectangle_Detection)

# 设置 OpenCV 所需的包
find_package(OpenCV 4.6 REQUIRED)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(OpenCV_Camera_Rectangle_Detection main.cpp)

# 链接 OpenCV 库
target_link_libraries(OpenCV_Camera_Rectangle_Detection ${OpenCV_LIBS})
