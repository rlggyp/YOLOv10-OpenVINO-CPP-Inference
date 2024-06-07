#include "inference.h"
#include "utils.h"

#include <iostream>
#include <opencv2/highgui.hpp>

int main(int argc, char **argv) {
	if (argc != 3) {
		std::cerr << "usage: " << argv[0] << " <model_path> <camera_index>" << std::endl;
		return 1;
	}

	const std::string model_path = argv[1];
	int camera_index = std::stoi(argv[2]);

	cv::VideoCapture capture(camera_index);

	if (!capture.isOpened()) {
		std::cerr << "ERROR: Could not open the camera" << std::endl;
		return 1;
	}

	const float confidence_threshold = 0.5;

	yolo::Inference inference(model_path, confidence_threshold);

	cv::Mat frame;

	const char escape_key = 27;

	while (true) {
		capture >> frame;

		if (frame.empty()) {
			std::cerr << "ERROR: Frame is empty" << std::endl;
			break;
		}

		std::vector<yolo::Detection> detections = inference.RunInference(frame);

		DrawDetectedObject(frame, detections);

		cv::imshow("camera", frame);

		if (cv::waitKey(1) == escape_key) {
			break;
		}
	}

	capture.release();
	cv::destroyAllWindows();

	return 0;
}
