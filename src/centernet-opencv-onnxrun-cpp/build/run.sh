# wsl 只有文本没有图像界面,需要安装 MobaXTerm 并在里面进入wsl并找到项目目录进行脚本执行
# cd /mnt/c/work/workspace/deeplearn-cpp/src/centernet-opencv-onnxrun-cpp/build/

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