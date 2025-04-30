#pragma once
#include <iostream>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>


/**
 * @brief Key point structure
*/
struct PosePoint
{
	int x;
	int y;
	float score;

	PosePoint()
	{
		x = 0;
		y = 0;
		score = 0.0;
	}
};

/**
 * @brief Detection box structure
*/
struct Box
{
	float x1;
	float y1;
	float x2;
	float y2;
	int cls;
	float conf;

	Box()
	{
		x1 = 0;
		y1 = 0;
		x2 = 0;
		y2 = 0;
		cls = 0;
		conf = 0;
	}
};

bool MixImage(cv::Mat& srcImage, cv::Mat mixImage, cv::Point startPoint);
std::tuple<cv::Mat, int, int> resize(cv::Mat& img, int w, int h);
bool compare_boxes(const Box& b1, const Box& b2);
float intersection_over_union(const Box& b1, const Box& b2);
std::vector<Box> non_maximum_suppression(std::vector<Box> boxes, float iou_thre);
cv::Mat img_cut(cv::Mat& image, int x1, int y1, int x2, int y2);
bool isEqual(float a, float b);
void draw_pose(cv::Mat& image, std::vector<std::vector<PosePoint>> points);
