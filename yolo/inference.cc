#include "inference.h"

#include <memory>

namespace yolo {
Inference::Inference(const std::string &model_path, const float &model_confidence_threshold) {
	model_input_shape_ = cv::Size(640, 640); // default size for dynamic model, prevent error
	model_confidence_threshold_ = model_confidence_threshold;
	InitialModel(model_path);
}

// if model dynamic, we need to set input shape
Inference::Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold) {
	model_input_shape_ = model_input_shape;
	model_confidence_threshold_ = model_confidence_threshold;

	InitialModel(model_path);
}

void Inference::InitialModel(const std::string &model_path) {
	ov::Core core;
	std::shared_ptr<ov::Model> model = core.read_model(model_path);

	short width, height;

	if (model->is_dynamic()) {
		model->reshape({1, 3, static_cast<long int>(model_input_shape_.height), static_cast<long int>(model_input_shape_.width)});
	} else {
		// get input shape from model
  	const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
  	const ov::Shape input_shape = inputs[0].get_shape();

		height = input_shape[1];
		width = input_shape[2];
		model_input_shape_ = cv::Size2f(width, height);
	}

  const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
  const ov::Shape output_shape = outputs[0].get_shape();

	height = output_shape[1];
	width = output_shape[2];
	model_output_shape_ = cv::Size(width, height);

	ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

  ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
  ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255, 255, 255 });
	ppp.input().model().set_layout("NCHW");
  ppp.output().tensor().set_element_type(ov::element::f32);

  model = ppp.build();
	compiled_model_ = core.compile_model(model, "AUTO");
	inference_request_ = compiled_model_.create_infer_request();
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
	float *detections = inference_request_.get_output_tensor().data<float>();
	const cv::Mat detection_outputs(model_output_shape_, CV_32F, (float *)detections);

	detections_.clear();

	// 0  1  2  3      4          5
	// x, y, w. h, confidence, class_id

	for (int i = 0; i < detection_outputs.rows; ++i) {
		double confidence = detection_outputs.at<float>(i, 4);

		if (confidence > model_confidence_threshold_) {
			const float x = detection_outputs.at<float>(i, 0);
			const float y = detection_outputs.at<float>(i, 1);
			const float w = detection_outputs.at<float>(i, 2);
			const float h = detection_outputs.at<float>(i, 3);

			Detection result;

			result.class_id = static_cast<short>(detection_outputs.at<float>(i, 5));
			result.confidence = confidence;
			result.box = GetBoundingBox(cv::Rect(x, y, w, h));

			detections_.push_back(result);
		}
	}
}

cv::Rect Inference::GetBoundingBox(const cv::Rect &src) {
	cv::Rect box = src;

	box.width = (box.width - box.x) * factor_.x;
	box.height = (box.height - box.y) * factor_.y;

	box.x *= factor_.x;
	box.y *= factor_.y;
	
	return box;
}
} // namespace yolo
