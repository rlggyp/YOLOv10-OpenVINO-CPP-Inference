#ifndef UTILS_H_
#define UTILS_H_

#include "inference.h"

void DrawDetectedObject(cv::Mat &frame, const std::vector<yolo::Detection> &detections, const std::vector<std::string> &class_names);
std::vector<std::string> GetClassNameFromMetadata(const std::string &metadata_path);

#endif // UTILS_H_
