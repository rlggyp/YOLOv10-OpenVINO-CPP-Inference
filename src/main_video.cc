#include "inference.h"
#include "utils.h"

#include <iostream>
#include <opencv2/highgui.hpp>

int main(const int argc, const char **argv) {
	if (argc != 3) {
		std::cerr << "usage: " << argv[0] << " <model_path> <video_path>" << std::endl;
		return 1;
	}

	const std::string model_path = argv[1];
	const std::string video_path = argv[2];

  const std::size_t pos = model_path.find_last_of("/");
	const std::string metadata_path = model_path.substr(0, pos + 1) + "metadata.yaml";
	const std::vector<std::string> class_names = GetClassNameFromMetadata(metadata_path);

	cv::VideoCapture capture(video_path);

	if (!capture.isOpened()) {
		std::cerr << "ERROR: Could not open the video file" << std::endl;
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

		DrawDetectedObject(frame, detections, class_names);

		cv::imshow("video", frame);

		if (cv::waitKey(10) == escape_key) {
			break;
		}
	}

	capture.release();
	cv::destroyAllWindows();

	return 0;
}
