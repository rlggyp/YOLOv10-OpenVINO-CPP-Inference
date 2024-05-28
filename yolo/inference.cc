#include "inference.h"

#include <memory>

namespace yolo {
Inference::Inference(const std::string &model_path) {
	model_score_threshold_ = 0.48;
	model_NMS_threshold_ = 0.48;
	InitialModel(model_path);
}

void Inference::InitialModel(const std::string &model_path) {
	ov::Core core;
	std::shared_ptr<ov::Model> model = core.read_model(model_path);
	ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

  ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
  ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255, 255, 255 });
	ppp.input().model().set_layout("NCHW");
  ppp.output().tensor().set_element_type(ov::element::f32);

  model = ppp.build();
	compiled_model_ = core.compile_model(model, "CPU");
	inference_request_ = compiled_model_.create_infer_request();

  const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
  const ov::Shape input_shape = inputs[0].get_shape();

	short height = input_shape[1];
	short width = input_shape[2];
	model_input_shape_ = cv::Size2f(width, height);
	std::cout << "model_input_shape: " << model_input_shape_ << std::endl;

  const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
  const ov::Shape output_shape = outputs[0].get_shape();

	height = output_shape[1];
	width = output_shape[2];
	model_output_shape_ = cv::Size(width, height);
	std::cout << "model_output_shape: " << model_output_shape_ << std::endl;
}

std::vector<Detection> Inference::RunInference(const cv::Mat &frame) {
	Preprocessing(frame);
	inference_request_.infer();
	PostProcessing();

	return detections_;
}

void Inference::Preprocessing(const cv::Mat &frame) {
	cv::resize(frame, resized_frame_, model_input_shape_, 0, 0, cv::INTER_AREA);

	factor_.x = static_cast<float>(frame.cols / model_input_shape_.width);
	factor_.y = static_cast<float>(frame.rows / model_input_shape_.height);

	float *input_data = (float *)resized_frame_.data;
	input_tensor_ = ov::Tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), input_data);
	inference_request_.set_input_tensor(input_tensor_);
}

void Inference::PostProcessing() {
	std::vector<int> class_list;
	std::vector<float> confidence_list;
	std::vector<cv::Rect> box_list;

	float *detections = inference_request_.get_output_tensor().data<float>();
	const cv::Mat detection_outputs(model_output_shape_, CV_32F, (float *)detections);
	// std::cout << "detection_outputs: " << detection_outputs << std::endl;

	// std::cout << std::endl << "lmao_: " << std::endl;
	// std::cout << detection_outputs.at<float>(0, 0) << ", ";
	// std::cout << detection_outputs.at<float>(0, 1) << ", ";
	// std::cout << detection_outputs.at<float>(0, 2) << ", ";
	// std::cout << detection_outputs.at<float>(0, 3) << ", ";
	// std::cout << detection_outputs.at<float>(0, 4) << ", ";
	// std::cout << detection_outputs.at<float>(0, 5) << std::endl;

	for (int i = 0; i < detection_outputs.rows; ++i) {
		// const cv::Mat classes_scores = detection_outputs.col(i).rowRange(4, detection_outputs.rows);

		 double score = detection_outputs.at<float>(i, 4);
		// double score = detection_outputs.at<float>(5, i);

		//cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id);

		if (score > model_score_threshold_) {
			//class_list.push_back(detection_outputs.at<float>(i, 6));
			class_list.push_back(detection_outputs.at<float>(i, 5));
			confidence_list.push_back(score);

			// const float x = detection_outputs.at<float>(i, 0);
			// const float y = detection_outputs.at<float>(i, 1);
			// const float w = detection_outputs.at<float>(i, 2);
			// const float h = detection_outputs.at<float>(i, 3);

			const float x = detection_outputs.at<float>(i, 0);
			const float y = detection_outputs.at<float>(i, 1);
			const float w = detection_outputs.at<float>(i, 2);
			const float h = detection_outputs.at<float>(i, 3);

			// std::cout << x << ", " << y << ", " << w  << ", " <<  h << ", " << score << ", " << detection_outputs.at<float>(i, 5) << std::endl;

			// const float x = detection_outputs.at<float>(0, i);
			// const float y = detection_outputs.at<float>(1, i);
			// const float w = detection_outputs.at<float>(2, i);
			// const float h = detection_outputs.at<float>(3, i);

			cv::Rect box;

			box.x = static_cast<int>(x);
			box.y = static_cast<int>(y);
			box.width = static_cast<int>(w);
			box.height = static_cast<int>(h);

			box_list.push_back(box);
		}
	}

	std::vector<int> NMS_result;
	cv::dnn::NMSBoxes(box_list, confidence_list, model_score_threshold_, model_NMS_threshold_, NMS_result);

	detections_.clear();

	for (int i = 0; i < NMS_result.size(); ++i) {
		Detection result;
		int id = NMS_result[i];

		result.class_id = class_list[id];
		result.confidence = confidence_list[id];
		result.box = GetBoundingBox(box_list[id]);

		detections_.push_back(result);
	}
}

cv::Rect Inference::GetBoundingBox(const cv::Rect &src) {
	cv::Rect box = src;

	box.width = (box.width - box.x) * factor_.x;
	box.height = (box.height -box.y) * factor_.y;

	box.x *= factor_.x;
	box.y *= factor_.y;
	
	return box;
}
} // namespace yolo
