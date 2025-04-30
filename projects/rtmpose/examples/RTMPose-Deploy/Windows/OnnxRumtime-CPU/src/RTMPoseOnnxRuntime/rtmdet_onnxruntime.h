#ifndef _RTM_DET_ONNX_RUNTIME_H_
#define _RTM_DET_ONNX_RUNTIME_H_

#include <string>

#include "opencv2/opencv.hpp"

#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include "rtmpose_utils.h"


class RTMDetOnnxruntime
{
public:
	RTMDetOnnxruntime() = delete;
	RTMDetOnnxruntime(const std::string& onnx_model_path);
	virtual~RTMDetOnnxruntime();

public:
	DetectBox Inference(const cv::Mat& input_mat);

private:
	void PrintModelInfo(Ort::Session& session);

private:
	Ort::Env m_env;
	Ort::Session m_session;

};

#endif // !_RTM_DET_ONNX_RUNTIME_H_
