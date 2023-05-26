#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "utils.h"



class RTMPose
{
public:
	RTMPose(std::string model_path, nvinfer1::ILogger &logger);
	void show();
	std::vector<PosePoint> predict(cv::Mat& image);
	~RTMPose();

private:
	static float input_w;
	static float input_h;
	static int extend_width;
	static int extend_height;
	static float mean[3];
	static float std[3];
	static int num_points;

	std::vector<int> offset;

	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;

	void* buffer[3];
	cudaStream_t stream;

	std::vector<float> preprocess(cv::Mat& image);
	std::vector<PosePoint> postprocess(std::vector<float> simcc_x_result, std::vector<float> simcc_y_result, int img_w, int img_h);
};
