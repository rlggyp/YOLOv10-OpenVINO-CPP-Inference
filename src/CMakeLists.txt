cmake_minimum_required(VERSION 3.10.2)
project(detect)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	/opt/intel/openvino/runtime/include
)

add_executable(${PROJECT_NAME} 
	main.cc
	inference.cc
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	/opt/intel/openvino/runtime/lib/intel64/libopenvino.so
)
