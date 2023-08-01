# cd /mnt/c/work/workspace/deeplearn-cpp/src/yolov8-face-landmarks-opencv-dnn/build/

# 编译
cmake ..
make clean
make -j8

# 删除中间产生的文件
rm -f CMakeCache.txt
rm -rf CMakeFiles
rm -f cmake_install.cmake
rm -f Makefile

# 运行可执行文件
./app