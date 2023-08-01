cmake_minimum_required(VERSION 3.4.1)
project(app)
find_package(OpenCV REQUIRED)

# 只有opencv依赖
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(app main.cpp)
target_link_libraries(app ${OpenCV_LIBS})
set_property(TARGET app PROPERTY CXX_STANDARD 14)

