#include "inference.h"
#include "utils.h"

#include <iostream>
#include <opencv2/highgui.hpp>

int main(int argc, char **argv) {
	if (argc != 3) {
		std::cerr << "usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
		return 1;
	}

	const std::string model_path = argv[1];
	const std::string image_path = argv[2];

  std::size_t pos = model_path.find_last_of("/");
	std::string metadata_path = model_path.substr(0, pos + 1) + "metadata.yaml";
	std::vector<std::string> class_names = GetClassNameFromMetadata(metadata_path);

	cv::Mat image = cv::imread(image_path);

	if (image.empty()) {
		std::cerr << "ERROR: image is empty" << std::endl;
		return 1;
	}

	const float confidence_threshold = 0.5;

	yolo::Inference inference(model_path, confidence_threshold);
	std::vector<yolo::Detection> detections = inference.RunInference(image);

	DrawDetectedObject(image, detections, class_names);
	imshow("image", image);

	const char escape_key = 27;

	while (cv::waitKey(0) != escape_key);

	cv::destroyAllWindows();

	return 0;
}
