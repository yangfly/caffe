cmake_minimum_required(VERSION 2.8.7)

set(source /home/yf/test/caffe/contrib/ssd/caffe/tools/convert_annoset.cpp)
message(STATUS "source=${source}")

get_filename_component(name ${source} NAME_WE)
get_filename_component(path ${source} PATH)
string(REPLACE "caffe/tools" "tutorial/tools" output ${path})


message(STATUS "name=${name}")
message(STATUS "path=${path}")
message(STATUS "output=${output}")