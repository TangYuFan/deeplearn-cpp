cmake_minimum_required(VERSION 3.4.1)
project(app)
find_package(OpenCV REQUIRED)
set(onnxruntime_INCLUDE_DIRS /opt/onnxruntime-linux-x64-1.6.0/include)
include_directories(${onnxruntime_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(app main.cpp)
target_link_libraries(app ${OpenCV_LIBS} /opt/onnxruntime-linux-x64-1.6.0/lib/libonnxruntime.so)
set_property(TARGET app PROPERTY CXX_STANDARD 14)



# 需要先安装opencv
# sudo apt-get install build-essential libgtk2.0-dev libgtk-3-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev
# 下载源码解压 https://codeload.github.com/opencv/opencv/zip/refs/tags/4.8.0
# cd opencv-4.8.0
# 编译安装
# mkdir build && cd build
# cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON ..
# make -j8 && make install

# 同时下载onnx库,注意cpu的和gpu的下载的不同的,另外上面set需要加入onnx的头文件目录,这个项目只支持1.6.0  https://github.com/microsoft/onnxruntime/releases
# cd /opt
# wget https://github.com/microsoft/onnxruntime/releases/download/v1.6.0/onnxruntime-linux-x64-1.6.0.tgz
# tar -xzvf onnxruntime-linux-x64-1.6.0.tgz
# rm -f onnxruntime-linux-x64-1.6.0.tgz
# 对应位置 /opt/onnxruntime-linux-x64-1.6.0/lib/libonnxruntime.so


# 如何判断 onnx版本？
# 模型可视化打开 ONNX v8 那么就对应了 1.8.0
# 模型可视化打开 ONNX v6 那么就对应了 1.6.0