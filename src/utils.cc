#include "utils.h"

#include <fstream>
#include <random>
#include <yaml-cpp/yaml.h>


void DrawDetectedObject(cv::Mat &frame, const std::vector<yolo::Detection> &detections, const std::vector<std::string> &class_names) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(120, 255);
	
	for (const auto &detection : detections) {
		const cv::Rect &box = detection.box;
		const float &confidence = detection.confidence;
		const int &class_id = detection.class_id;
		
		const cv::Scalar color = cv::Scalar(dis(gen), dis(gen), dis(gen));
		cv::rectangle(frame, box, color, 3);

		std::string class_string;

		if (class_names.empty())
			class_string = "id[" + std::to_string(class_id) + "] " + std::to_string(confidence).substr(0, 4);
		else
			class_string = class_names[class_id] + " " + std::to_string(confidence).substr(0, 4);
		
		const cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, 0);
		const cv::Rect text_box(box.x - 2, box.y - 27, text_size.width + 10, text_size.height + 15);
		
		cv::rectangle(frame, text_box, color, cv::FILLED);
		cv::putText(frame, class_string, cv::Point(box.x + 5, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2, 0);
	}
}

std::vector<std::string> GetClassNameFromMetadata(const std::string &metadata_path) {
	std::ifstream check_file(metadata_path);

	if (!check_file.is_open()) {
		std::cerr << "Unable to open file: " << metadata_path << std::endl;
		return {};
	}

	check_file.close();

	YAML::Node metadata = YAML::LoadFile(metadata_path);
	std::vector<std::string> class_names;

	if (!metadata["names"]) {
		std::cerr << "ERROR: 'names' node not found in the YAML file" << std::endl;
		return {};
	}

	for (int i = 0; i < metadata["names"].size(); ++i) {
		std::string class_name = metadata["names"][std::to_string(i)].as<std::string>();
		class_names.push_back(class_name);
	}

	return class_names;
}
