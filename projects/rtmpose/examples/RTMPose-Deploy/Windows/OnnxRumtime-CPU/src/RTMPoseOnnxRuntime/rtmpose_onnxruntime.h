#ifndef _RTM_POSE_ONNXRUNTIME_H_
#define _RTM_POSE_ONNXRUNTIME_H_

#include <string>

#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include "opencv2/opencv.hpp"

#include "rtmdet_onnxruntime.h"
#include "rtmpose_utils.h"

class RTMPoseOnnxruntime
{
public:
	RTMPoseOnnxruntime() = delete;
	RTMPoseOnnxruntime(const std::string& onnx_model_path);
	virtual~RTMPoseOnnxruntime();

public:
	std::vector<PosePoint> Inference(const cv::Mat& input_mat, const DetectBox& box);

private:
	std::pair<cv::Mat, cv::Mat> CropImageByDetectBox(const cv::Mat& input_image, const DetectBox& box);

private:
	void PrintModelInfo(Ort::Session& session);

private:
	Ort::Env m_env;
	Ort::Session m_session;
};

#endif // !_RTM_POSE_ONNXRUNTIME_H_
