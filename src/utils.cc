#include "utils.h"

#include <random>

void DrawDetectedObject(cv::Mat &frame, const std::vector<yolo::Detection> &detections) {
	for (int i = 0; i < detections.size(); ++i) {
		yolo::Detection detection = detections[i];
		cv::Rect box = detection.box;
		float confidence = detection.confidence;
		int class_id = detection.class_id;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(120, 255);
		cv::Scalar color = cv::Scalar(dis(gen),
				dis(gen),
				dis(gen));
		cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), color, 3);

		std::string classString = "id(" + std::to_string(class_id) + ')' + std::to_string(confidence).substr(0, 4);
		cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 2, 0);
		cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
		cv::rectangle(frame, textBox, color, cv::FILLED);
		cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 2, 0);
	}
}
