#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "utils.h"



class RTMDet
{
public:
	RTMDet(std::string model_path, nvinfer1::ILogger& logger, float conf_thre=0.5, float iou_thre=0.65);
	void show();
	std::vector<Box> predict(cv::Mat& image);
	~RTMDet();

private:
	static float input_w;
	static float input_h;
	static float mean[3];
	static float std[3];

	float conf_thre;
	float iou_thre;
	std::vector<int> offset;

	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;

	void* buffer[2];
	cudaStream_t stream;

	std::vector<float> preprocess(cv::Mat& image);
	std::vector<Box> postprocess(std::vector<float> boxes_result, int img_w, int img_h);
};
