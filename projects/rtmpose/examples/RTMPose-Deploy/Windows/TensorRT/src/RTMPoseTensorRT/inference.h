#pragma once
#include <iostream>
#include <string>

#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "rtmdet.h"
#include "rtmpose.h"
#include "utils.h"



std::vector<std::vector<PosePoint>> inference(cv::Mat& image, RTMDet& detect_model, RTMPose& pose_model);
