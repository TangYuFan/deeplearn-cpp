cmake_minimum_required(VERSION 3.4.1)
project(app)
find_package(OpenCV REQUIRED)
set(onnxruntime_INCLUDE_DIRS /opt/onnxruntime-linux-x64-1.6.0/include)

# 自定义头文件位置
include_directories(.)
# onnx头文件位置
include_directories(${onnxruntime_INCLUDE_DIRS})
# opencn头文件位置
include_directories(${OpenCV_INCLUDE_DIRS})

# main函数之外用到的自定义源文件也需要加在后面
add_executable(app main.cpp text_rec.cpp text_det.cpp text_angle_cls.cpp)


target_include_directories(app PUBLIC .)
target_link_libraries(app ${OpenCV_LIBS} /opt/onnxruntime-linux-x64-1.6.0/lib/libonnxruntime.so)
set_property(TARGET app PROPERTY CXX_STANDARD 14)
