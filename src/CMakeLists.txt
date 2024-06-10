cmake_minimum_required(VERSION 3.10.2)
project(YOLOv10_OpenVINO)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} /opt/intel/openvino/runtime/include)

set(SOURCES
	utils.cc
	inference.cc
)

add_library(inference_lib ${SOURCES})
target_link_libraries(inference_lib ${OpenCV_LIBS} /opt/intel/openvino/runtime/lib/intel64/libopenvino.so yaml-cpp)

add_executable(camera main_camera.cc)
add_executable(detect main_detect.cc)
add_executable(video main_video.cc)

target_link_libraries(camera inference_lib)
target_link_libraries(detect inference_lib)
target_link_libraries(video inference_lib)
