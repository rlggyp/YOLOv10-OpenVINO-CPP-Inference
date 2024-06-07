cmake_minimum_required(VERSION 3.10.2)
project(YOLOv10_OpenVINO)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	/opt/intel/openvino/runtime/include
)

set(SOURCES
	utils.cc
	inference.cc
)

add_executable(camera main_camera.cc ${SOURCES})
add_executable(detect main_detect.cc ${SOURCES})
add_executable(video main_video.cc ${SOURCES})

target_link_libraries(camera ${OpenCV_LIBS} /opt/intel/openvino/runtime/lib/intel64/libopenvino.so)
target_link_libraries(detect ${OpenCV_LIBS} /opt/intel/openvino/runtime/lib/intel64/libopenvino.so)
target_link_libraries(video ${OpenCV_LIBS} /opt/intel/openvino/runtime/lib/intel64/libopenvino.so)
